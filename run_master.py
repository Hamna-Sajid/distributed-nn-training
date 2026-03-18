"""Entry point to start the master node."""
from coordination.master import Master

if __name__ == "__main__":
    master = Master(n_workers=2, n_epochs=20, lr=0.01)
    master.run()