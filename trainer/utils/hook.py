import sys
import logging

logger = logging.getLogger(__name__)

_orig_except_hook = None


def _global_except_hook(exctype, value, traceback):
    """Catches an unhandled exception and call MPI_Abort()."""
    try:
        if _orig_except_hook:
            _orig_except_hook(exctype, value, traceback)
        else:
            sys.__excepthook__(exctype, value, traceback)

    finally:
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        logger.warning("******************************************")
        logger.warning("DefaultTrainer:")
        logger.warning(f"   Uncaught exception on rank {rank}.")
        logger.warning("   Calling MPI_Abort() to shut down MPI...")
        logger.warning("******************************************")
        logging.shutdown()

        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            # Something is completely broken...
            # There's nothing we can do any more
            sys.stderr.write("Sorry, failed to stop MPI and the process may hang.\n")
            sys.stderr.flush()
            raise e


def add_hook():
    """
    Add a global hook function that captures all unhandled exceptions.
    The function calls MPI_Abort() to force all processes abort.

    An MPI runtime is expected to kill all of its child processes
    if one of them exits abnormally or without calling `MPI_Finalize()`.
    However, when a Python program run on `mpi4py`, the MPI runtime
    often fails to detect a process failure, and the rest of the processes
    hang infinitely.

    See https://github.com/chainer/chainermn/issues/236 and
    https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html for more
    information.
    """
    global _orig_except_hook

    if _orig_except_hook is not None:
        logger.warning("GlobalExceptHook.add_hook() seems to be called multiple times. Ignoring.")
        return

    logger.info("Adding global except hook for the distributed job to shutdown MPI if unhandled exception is raised on some of the ranks.")
    _orig_except_hook = sys.excepthook
    sys.excepthook = _global_except_hook
