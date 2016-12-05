#include <iostream>
#include <petscdmnetwork.h>
#include <memory>

static char help[] = "This example demonstrates the use of DMNetwork in a heat transfer problem. \n\n";
using namespace std;

class Wall {
public:
	Wall(double L, double alpha, int nx) 
		: L(L), alpha(alpha), nx(nx) {}

	double getLenght() {
		return this->L;
	}

	double getAlpha() {
		return this->alpha;
	}

	int getNx() {
		return this->nx;
	}

private:
	double L;
	double alpha;
	int nx;
};


enum NodeType {
	JUNCTION,
	BOUNDARY,
};

class Node {
public:
	Node(NodeType type, double value=0.0) 
		: type(type), value(value) {}

	NodeType getType() {
		return type;
	}

	double getValue() {
		return value;
	}

private:
	NodeType type;
	double value;
};

void create_network_layout(DM network) {
	PetscInt edge_list[] = {
		0, 2,
		2, 1
	};

	PetscInt n_vertices = 3;
	PetscInt n_edges = 2;
	
	DMNetworkSetSizes(network, n_vertices, n_edges, PETSC_DETERMINE, PETSC_DETERMINE);
	DMNetworkSetEdgeList(network, edge_list);
	DMNetworkLayoutSetUp(network);
}

void add_network_components(DM network) {
	PetscInt eStart, eEnd, vStart, vEnd, wall_key, node_key;

	DMNetworkGetEdgeRange(network, &eStart, &eEnd);
	DMNetworkGetVertexRange(network, &vStart, &vEnd);

	DMNetworkRegisterComponent(network, "wall", sizeof(Wall), &wall_key);
	DMNetworkRegisterComponent(network, "node", sizeof(Node), &node_key);

	int ref_nx = 4;
	double ref_length = 1.0;
	double ref_alpha = 1.0;

	for (int e = eStart; e < eEnd; e++) {
		shared_ptr<Wall> pipe(new Wall(ref_length, ref_alpha, ref_nx));
		DMNetworkAddComponent(network, e, wall_key, pipe.get());
		DMNetworkAddNumVariables(network, e, 2 * pipe->getNx());
	}

	shared_ptr<Node> node_0(new Node(BOUNDARY, 1.0));
	shared_ptr<Node> node_1(new Node(BOUNDARY, 0.0));
	shared_ptr<Node> node_2(new Node(JUNCTION));

	DMNetworkAddComponent(network, vStart + 0, node_key, node_0.get());
	DMNetworkAddNumVariables(network, vStart + 0, 1);

	DMNetworkAddComponent(network, vStart + 1, node_key, node_1.get());
	DMNetworkAddNumVariables(network, vStart + 1, 1);
	
	DMNetworkAddComponent(network, vStart + 2, node_key, node_2.get());
	DMNetworkAddNumVariables(network, vStart + 2, 1);
}

DM create_global_network() {
	DM network;
	DMNetworkCreate(PETSC_COMM_WORLD, &network);

	create_network_layout(network);
	add_network_components(network);

	DMSetUp(network);

	return network;
}

int main(int argc, char ** argv) {
	PetscErrorCode ierr;
	PetscInt rank, size;

	PetscInitialize(&argc, &argv, (char*)0, help);
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);

	DM network = create_global_network();

	if (size > 1) {
		DM distnetworkdm;
		ierr = DMNetworkDistribute(network, 0, &distnetworkdm); CHKERRQ(ierr);
		ierr = DMDestroy(&network); CHKERRQ(ierr);
		network = distnetworkdm;
	}

	DMDestroy(&network);

	PetscFinalize();
	return 0;
}