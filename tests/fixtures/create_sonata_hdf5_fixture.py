import pathlib
import sys

import h5py


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: create_sonata_hdf5_fixture.py <fixture-root>", file=sys.stderr)
        return 2

    root = pathlib.Path(sys.argv[1])
    networks = root / "networks"
    networks.mkdir(parents=True, exist_ok=True)

    with h5py.File(networks / "nodes.h5", "w") as h5:
        v1 = h5.create_group("/nodes/v1")
        v1.create_dataset("node_id", data=[0, 1])
        v1.create_dataset("node_type_id", data=[1, 1])

        v2 = h5.create_group("/nodes/v2")
        v2.create_dataset("node_id", data=[0, 1])
        v2.create_dataset("node_type_id", data=[2, 2])

    with h5py.File(networks / "edges.h5", "w") as h5:
        edges = h5.create_group("/edges/v1_to_v2")
        edges.create_dataset("source_node_id", data=[0, 1])
        edges.create_dataset("target_node_id", data=[0, 1])
        edges.create_dataset("weight", data=[0.5, 0.75])
        edges.create_dataset("delay", data=[1.0, 2.0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

