#include <iostream>
#include <memory>
#include <vector>

#include <petscdmnetwork.h>
#include <petscts.h>
#include <petscdmda.h>

static char help[] = "This example demonstrates the use of DMNetwork in a heat transfer problem. \n\n";
using namespace std;

#define CHECK_DOMAIN_DECOMPOSITION 0

struct Wall {
    int index;
	double L;
	double alpha;
	int nx;
    DM da;
};


enum NodeType {
	JUNCTION,
	BOUNDARY,
};

struct Node {
    int index;
	NodeType type;
	double value;
};

struct Domain {
    DM network;
    std::vector<shared_ptr<Wall>> walls;
    std::vector<shared_ptr<Node>> nodes;
    Vec local_x;
    Vec local_x_dot;
};

void create_network_layout(DM network) {
    PetscInt *edge_list = NULL;
    PetscInt n_vertices=0, n_edges=0, rank;


    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (!rank) {
        PetscMalloc1(4, &edge_list);
        edge_list[0] = 0; edge_list[1] = 2;
        edge_list[2] = 2; edge_list[3] = 1;

        n_vertices = 3;
        n_edges = 2;
    }
	
	DMNetworkSetSizes(network, n_vertices, n_edges, PETSC_DETERMINE, PETSC_DETERMINE);
    DMNetworkSetEdgeList(network, edge_list);
    DMNetworkLayoutSetUp(network);

    if (!rank) {
        PetscFree(edge_list);
    }
}

void add_network_components(shared_ptr<Domain> domain) {
	PetscInt eStart, eEnd, vStart, vEnd, wall_key, node_key;

	DMNetworkGetEdgeRange(domain->network, &eStart, &eEnd);
	DMNetworkGetVertexRange(domain->network, &vStart, &vEnd);

	DMNetworkRegisterComponent(domain->network, "wall", sizeof(Wall), &wall_key);
	DMNetworkRegisterComponent(domain->network, "node", sizeof(Node), &node_key);

    PetscInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank) {
        return;
    }

	int ref_nx = 4;
	double ref_length = 1.0;
	double ref_alpha = 1.0;

	for (int e = eStart; e < eEnd; e++) {
		shared_ptr<Wall> wall(new Wall);
        wall->L = ref_length;
        wall->alpha = ref_alpha;
        wall->nx = ref_nx;
        wall->index = e - eStart;
		DMNetworkAddComponent(domain->network, e, wall_key, wall.get());
		DMNetworkAddNumVariables(domain->network, e, wall->nx);
        domain->walls.push_back(wall);
	}

	shared_ptr<Node> node_0(new Node);
    node_0->index = 0;
    node_0->type = BOUNDARY;
    node_0->value = 1.0;
	shared_ptr<Node> node_1(new Node);
    node_1->index = 1;
    node_1->type = BOUNDARY;
    node_1->value = 0.0;
	shared_ptr<Node> node_2(new Node);
    node_2->index = 2;
    node_2->type = JUNCTION;
    node_2->value = 0.0;

	DMNetworkAddComponent(domain->network, vStart + 0, node_key, node_0.get());
	DMNetworkAddNumVariables(domain->network, vStart + 0, 1);
    domain->nodes.push_back(node_0);

	DMNetworkAddComponent(domain->network, vStart + 1, node_key, node_1.get());
	DMNetworkAddNumVariables(domain->network, vStart + 1, 1);
    domain->nodes.push_back(node_1);

	DMNetworkAddComponent(domain->network, vStart + 2, node_key, node_2.get());
	DMNetworkAddNumVariables(domain->network, vStart + 2, 1);
    domain->nodes.push_back(node_2);
}

shared_ptr<Domain> create_domain() {
    shared_ptr<Domain> domain(new Domain);
	DMNetworkCreate(PETSC_COMM_WORLD, &domain->network);

    create_network_layout(domain->network);

	add_network_components(domain);

	DMSetUp(domain->network);

	return domain;
}

void set_up_domain(shared_ptr<Domain> domain) {
    DM network = domain->network;

    DMNetworkComponentGenericDataType *nwarr;
    DMNetworkGetComponentDataArray(network, &nwarr);

    PetscInt edge_start, edge_end;
    DMNetworkGetEdgeRange(network, &edge_start, &edge_end);

    for (int e = edge_start; e < edge_end; e++) {
        PetscInt offset;
        DMNetworkGetComponentTypeOffset(network, e, 0, NULL, &offset);

        Wall* wall = (Wall*)(nwarr + offset);

        DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_GHOSTED, wall->nx, 1, 1, PETSC_NULL, &(wall->da));
        DMDASetUniformCoordinates(wall->da, 0, wall->L, 0, 0, 0, 0);
    }
}

#undef __FUNCT__
#define __FUNCT__ "SolveTimeStep"
PetscErrorCode SolveTimeStep(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void* ctx) {
    PetscInt rank, size;
    DM network;
    Vec localF;
    Domain* domain = (Domain*)ctx;
    const PetscScalar *xarr, *xdotarr;
    PetscScalar *farr;
    PetscInt eStart, eEnd, vStart, vEnd;
    DMNetworkComponentGenericDataType *nwarr;
    PetscErrorCode ierr;
    PetscInt vfrom, vto;
    const PetscInt *cone;
    PetscScalar *wall_x, *wall_x_dot, *wall_f;

    PetscFunctionBegin;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    VecSet(F, 0.0);

    TSGetDM(ts, &network);
    DMGetLocalVector(network, &localF);
    VecSet(localF, 0.0);

    // Update ghost cells
    DMGlobalToLocalBegin(network, X, INSERT_VALUES, domain->local_x);
    DMGlobalToLocalEnd(network, X, INSERT_VALUES, domain->local_x);

    int local_size, global_size;
    VecGetLocalSize(domain->local_x, &local_size);
    VecGetSize(domain->local_x, &global_size);

    VecGetArrayRead(domain->local_x, &xarr);
    VecGetArrayRead(domain->local_x_dot, &xdotarr);
    VecGetArray(localF, &farr);

    DMNetworkGetEdgeRange(network, &eStart, &eEnd);
    DMNetworkGetComponentDataArray(network, &nwarr);

    for (int e = eStart; e < eEnd; e++) {
        PetscInt component_offset;
        PetscInt var_offset;
        PetscInt vfrom_offset;
        PetscInt vto_offset;
        
        DMNetworkGetComponentTypeOffset(network, e, 0, NULL, &component_offset);
        DMNetworkGetVariableOffset(network, e, &var_offset);

        wall_x = (PetscScalar*)(xarr + var_offset);
        wall_x_dot = (PetscScalar*)(xdotarr + var_offset);
        wall_f = (PetscScalar*)(farr + var_offset);
        Wall* wall = (Wall*)(nwarr + component_offset);

        ierr = DMNetworkGetConnectedNodes(network, e, &cone); CHKERRQ(ierr);
        vfrom = cone[0];
        vto = cone[1];

        PetscInt start, n;
        DMDAGetCorners(wall->da, &start, NULL, NULL, &n, NULL, NULL);

        PetscReal dx = wall->L / n;

        // Interior points
        for (int i = start + 1; i < start + n - 1; i++) {
            double vc_inlet_flux = -wall->alpha * (wall_x[i] - wall_x[i - 1]) / dx;
            double vc_outlet_flux = -wall->alpha * (wall_x[i + 1] - wall_x[i]) / dx;

            wall_f[i] += vc_outlet_flux - vc_inlet_flux;

            wall_f[i] += wall_x_dot[i] * dx;
        }

        // Boundary conditions
        ierr = DMNetworkGetVariableOffset(network, vfrom, &vfrom_offset); CHKERRQ(ierr);
        ierr = DMNetworkGetVariableOffset(network, vto, &vto_offset); CHKERRQ(ierr);

        DMNetworkGetComponentTypeOffset(network, vfrom, 0, NULL, &component_offset);
        Node* node_from = (Node*)(nwarr + component_offset);
        DMNetworkGetComponentTypeOffset(network, vto, 0, NULL, &component_offset);
        Node* node_to = (Node*)(nwarr + component_offset);

        wall_f[start] += -wall->alpha * (wall_x[start + 1] - wall_x[start]) / dx + wall_x_dot[start] * dx;
        wall_f[start + n - 1] += +wall->alpha * (wall_x[start + n - 1] - wall_x[start + n - 2]) / dx + wall_x_dot[start + n - 1] * dx;

        double inlet_flux = -wall->alpha * (wall_x[start] - (xarr + vfrom_offset)[0]) / (dx / 2.0);
        double outlet_flux = -wall->alpha * ((xarr + vto_offset)[0] - wall_x[start + n - 1]) / (dx / 2.0);

        wall_f[start] -= inlet_flux;
        wall_f[start + n - 1] += outlet_flux;

        if (node_from->type == JUNCTION) {
            (farr + vfrom_offset)[0] += inlet_flux;
        }
        if (node_to->type == JUNCTION) {
            (farr + vto_offset)[0] -= outlet_flux;
        }
    }

    DMNetworkGetVertexRange(network, &vStart, &vEnd);
    for (int v = vStart; v < vEnd; v++) {
        PetscInt component_offset, var_offset;
        DMNetworkGetComponentTypeOffset(network, v, 0, NULL, &component_offset);
        DMNetworkGetVariableOffset(network, v, &var_offset);
                
        Node* node = (Node*)(nwarr + component_offset);
        
        PetscBool is_ghost;
        DMNetworkIsGhostVertex(network, v, &is_ghost);

        if (is_ghost || node->type != BOUNDARY) {
            continue;
        }

        (farr + var_offset)[0] = node->value - (xarr + var_offset)[0];
    }

    VecRestoreArrayRead(domain->local_x, &xarr);
    VecRestoreArrayRead(domain->local_x_dot, &xdotarr);
    VecRestoreArray(localF, &farr);

    DMLocalToGlobalBegin(network, localF, ADD_VALUES, F);
    DMLocalToGlobalEnd(network, localF, ADD_VALUES, F);
    DMRestoreLocalVector(network, &localF);

    PetscFunctionReturn(0);
}

int main(int argc, char ** argv) {
	PetscErrorCode ierr;
	PetscInt rank, size;

	PetscInitialize(&argc, &argv, (char*)0, help);
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);

    shared_ptr<Domain> domain = create_domain();

	if (size > 1) {
        PetscOptionsSetValue(NULL, "-petscpartitioner_type", "simple");
        if (rank == 0) cout << "Distributing" << endl;
		DM distnetworkdm;
		ierr = DMNetworkDistribute(domain->network, 0, &distnetworkdm); CHKERRQ(ierr);
		ierr = DMDestroy(&domain->network); CHKERRQ(ierr);
        domain = shared_ptr<Domain>(new Domain);
		domain->network = distnetworkdm;
	}

    set_up_domain(domain);
    DM network = domain->network;

#if CHECK_DOMAIN_DECOMPOSITION
    DMNetworkComponentGenericDataType *nwarr;
    DMNetworkGetComponentDataArray(network, &nwarr);

    for (int prc = 0; prc < size; prc++) {
        MPI_Barrier(PETSC_COMM_WORLD);
        
        if (prc != rank) {
            continue;
        }

        cout << "prc " << rank << endl;

        int start, end;

        DMNetworkGetEdgeRange(network, &start, &end);
        cout << "start " << start << " end " << end << endl;
        for (int i = start; i < end; i++) {
            PetscInt edge_offset;
            DMNetworkGetComponentTypeOffset(network, i, 0, NULL, &edge_offset);

            Wall* wall = (Wall*)(nwarr + edge_offset);
            std::cout << "Wall " << wall->index << std::endl;
        }

        DMNetworkGetVertexRange(network, &start, &end);

        std::cout << "vertex start " << start << ", end " << end  << ", count " << end-start << std::endl;

        for (int i = start; i < end; i++) {
            PetscInt vertex_offset;
            DMNetworkGetComponentTypeOffset(network, i, 0, NULL, &vertex_offset);

            Node* node = (Node*)(nwarr + vertex_offset);
            PetscBool is_ghost;
            DMNetworkIsGhostVertex(network, i, &is_ghost);
            std::cout << "Node " << node->index << " type " << ((node->type == JUNCTION) ? "junction" : "boundary") << " ghost: " << (is_ghost ? "true": "false") << std::endl;
        }
    }
#endif

    
    //DMNetworkHasJacobian(network, PETSC_TRUE, PETSC_TRUE);
    
    DMCreateLocalVector(network, &(domain->local_x));
    DMCreateLocalVector(network, &(domain->local_x_dot));

    TS ts;
    TSCreate(PETSC_COMM_WORLD, &ts);
    TSSetDM(ts, network);
    TSSetType(ts, TSBEULER); 
    TSSetIFunction(ts, NULL, SolveTimeStep, domain.get());
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);

    PetscScalar final_time = 1.0;
    TSSetDuration(ts, -1, final_time);
    TSSetInitialTimeStep(ts, 0.0, 0.5);

    TSSetFromOptions(ts);

    Vec x;
    DMCreateGlobalVector(domain->network, &x);
    VecSet(x, 0.0);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    
    TSSolve(ts, x);

    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	DMDestroy(&network);
    TSDestroy(&ts);

	PetscFinalize();
	return 0;
}