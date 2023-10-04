import logging
from mpi4py import MPI
import os
import re
import subprocess
import torch

logger = logging.getLogger(__name__)


class MPIAdapter:
    """
    MPIAdapter automatically detects and analyzes the training environment for distributed training
    and offers methods to set up distributed training jobs.

    For example, it determines whether training happens on AML, Philly, or locally.
    It also determines variables such as the world size and the rank of each GPU.
    """

    def __init__(self, port='55551', set_env_vars=True):
        local_address = '127.0.0.1'
        default_torch_distributed_port = port  # chosen arbitrarily

        if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
            # application was started without MPI
            # default to single node with single process
            self.env_info = 'no MPI'
            self.world_size = 1
            self.local_size = 1
            self.rank = 0
            self.local_rank = 0
            self.master_address = local_address
            self.master_port = default_torch_distributed_port
        else:
            # application was started with MPI
            # get MPI parameters
            self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
            self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

            if 'PHILLY_CONTAINER_IP' in os.environ:
                # application is running on Philly
                # read environment variables on master node and broadcast via MPI
                self.env_info = 'philly'
                if self.rank == 0:
                    self.master_address = os.environ['PHILLY_CONTAINER_IP']
                    self.master_port = os.environ['PHILLY_CONTAINER_PORT_RANGE_START']
                else:
                    self.master_address = None
                    self.master_port = None
                self.master_address = MPI.COMM_WORLD.bcast(self.master_address, root=0)
                self.master_port = MPI.COMM_WORLD.bcast(self.master_port, root=0)
            elif "AMLK8S_NUM_WORKER" in os.environ or "AZ_CMK8S_JOB_WORK_DIR" in os.environ:
                # application is running on AMLK8S (ITP)
                # read master address from a specific file.
                self.env_info = 'AMLK8S (ITP)'
                # from: https://k8s-wiki.azureml.com/faq.html
                regexp = r"[\s\S]*export[\s]*DLTS_SD_worker0_IP=([0-9.]+)[\s|s]*"
                with open("/dlts-runtime/env/init.env", 'r') as f:
                    line = f.read()
                match = re.match(regexp, line)
                if match:
                    self.master_address = str(match.group(1))
                else:
                    # Did not find master node ip in file. It must be a single-node
                    # debugging job with custom "mpirun" command
                    assert self.world_size == self.local_size, \
                        "It's not a single-node debugging job on AMLK8S (ITP), but no master ip is found in file."
                    self.env_info = 'single-node AMLK8S (ITP) debugging job'
                    self.master_address = local_address
                self.master_port = default_torch_distributed_port
            elif 'AZ_BATCH_MASTER_NODE' in os.environ:
                # application is running on multiple nodes on AML
                self.env_info = 'multi-node AML'
                master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
                self.master_address = master_node_params[0]
                self.master_port = default_torch_distributed_port
            elif self.world_size == self.local_size:
                # application is running with MPI on single node
                self.env_info = 'single-node AML or other MPI environment'
                self.master_address = local_address
                self.master_port = default_torch_distributed_port
            else:
                # multi-node MPI environment, but not Philly or AML
                # we use "hostname -I" command on rank 0 to get the master address
                self.env_info = 'multi-node other MPI environment'
                if self.rank == 0:
                    hostname_cmd = ["hostname -I"]
                    result = subprocess.check_output(hostname_cmd, shell=True)
                    self.master_address = result.decode('utf-8').split()[0]
                    self.master_port = default_torch_distributed_port
                else:
                    self.master_address = None
                    self.master_port = None
                self.master_address = MPI.COMM_WORLD.bcast(self.master_address, root=0)
                self.master_port = MPI.COMM_WORLD.bcast(self.master_port, root=0)

        self.init_method_url = f'tcp://{self.master_address}:{self.master_port}'
        if set_env_vars:
            self._set_env_vars()

    def log_info(self):
        """
        Logs information about distributed training environment.
        """
        # of not printing logger.info messages on processes with rank > 0
        logger.warning('----------------')
        logger.warning('MPI Adapter data')
        logger.warning('----------------')
        logger.warning(f'environment info: {self.env_info}')
        logger.warning(f'init method url: {self.init_method_url}')
        logger.warning(f'world size: {self.world_size}')
        logger.warning(f'local size: {self.local_size}')
        logger.warning(f'rank: {self.rank}')
        logger.warning(f'local rank: {self.local_rank}')
        logger.warning(f'master address: {self.master_address}')
        logger.warning(f'master port: {self.master_port}')
        logger.warning('----------------')

    def init_process_group(self, backend):
        """
        Initializes the default PyTorch distributed process group.
        """
        # of not printing logger.info messages on processes with rank > 0
        logger.warning('trying to initialize process group ...')
        torch.distributed.init_process_group(backend=backend,
                                             init_method=self.init_method_url,
                                             world_size=self.world_size,
                                             rank=self.rank)
        logger.warning('process group initialized')

    def _set_env_vars(self):
        """
        Sets environment variables for world size, rank, local rank, master addr, and master port.
        """
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ['MASTER_ADDR'] = self.master_address
        os.environ['MASTER_PORT'] = self.master_port
