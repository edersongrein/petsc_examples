static char help[] = "This example demonstrate the use of DMNetwork to solve a pipe flow single phase network problem";

#include <petscdmnetwork.h>

struct _p_VERTEXDATA {
	char		bc_type[20];
	PetscScalar bc_value;
	PetscInt	index;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct _p_VERTEXDATA *VERTEXDATA;


struct _p_PIPEDATA {
	PetscScalar diameter;
	PetscScalar length;
	PetscInt	nx;
	PetscInt	inlet_vertex;
	PetscInt	outlet_vertex;
	PetscInt	index;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct _p_PIPEDATA *PIPEDATA;

int main(int argc, char **argv) {
	PetscInt							rank, n_prcs;
	PetscErrorCode						ierr;
	DM									networkdm;
	PetscInt							componentkeys[2];
	DMNetworkComponentGenericDataType	*components_data;
	PetscInt							*connectivity;
	PetscInt							edges_count = 0;
	PetscInt							vertices_count = 0;
	PetscInt							eStart, eEnd, vStart, vEnd;
	PIPEDATA							*pipes_data;
	VERTEXDATA							*vertices_data;
	PetscInt							key, offset;
	DM									composite;

	PetscInitialize(&argc, &argv, NULL, help);
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	/* Create an empty network object */
	ierr = DMNetworkCreate(PETSC_COMM_WORLD, &networkdm); CHKERRQ(ierr);

	// Registering components
	ierr = DMNetworkRegisterComponent(networkdm, "pipe", sizeof(struct _p_PIPEDATA), &componentkeys[0]); CHKERRQ(ierr);
	ierr = DMNetworkRegisterComponent(networkdm, "vertex", sizeof(struct _p_VERTEXDATA), &componentkeys[1]); CHKERRQ(ierr);

	if (rank == 0) {
		ierr = PetscMalloc1(4, &connectivity); CHKERRQ(ierr);
		connectivity[0] = 0;
		connectivity[1] = 2;
		connectivity[2] = 2;
		connectivity[3] = 1;
		edges_count = 2;
		vertices_count = 3;
	}

	// Size
	ierr = DMNetworkSetSizes(networkdm, vertices_count, edges_count, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);

	// Connectivity
	ierr = DMNetworkSetEdgeList(networkdm, connectivity); CHKERRQ(ierr);
	ierr = DMNetworkLayoutSetUp(networkdm); CHKERRQ(ierr);

	// Components 
	if (rank == 0) {
		PetscMalloc1(vertices_count, &vertices_data);
		PetscMalloc1(edges_count, &pipes_data);

		for (int i = 0; i < vertices_count; i++) {
			PetscMalloc1(1, &(vertices_data[i]));
		}
		for (int i = 0; i < edges_count; i++) {
			PetscMalloc1(1, &(pipes_data[i]));
		}

		strcpy(vertices_data[0]->bc_type, "flow_rate");
		vertices_data[0]->bc_value = 0.01;
		vertices_data[0]->index = 0;

		strcpy(vertices_data[1]->bc_type, "pressure");
		vertices_data[1]->bc_value = 1.0e5;
		vertices_data[1]->index = 1;

		for (int i = 2; i < vertices_count; i++) {
			vertices_data[i]->index = i;
		}

		for (int i = 0; i < edges_count; i++) {
			pipes_data[i]->diameter = 0.1;
			pipes_data[i]->length = 100.0;
			pipes_data[i]->nx = 4;
			pipes_data[i]->inlet_vertex = connectivity[2 * i];
			pipes_data[i]->outlet_vertex = connectivity[2 * i + 1];
			pipes_data[i]->index = i;
		}
	}

	DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd);
	for (int i = eStart; i < eEnd; i++) {
		DMNetworkAddComponent(networkdm, i, componentkeys[0], pipes_data[i - eStart]);
		DMNetworkAddNumVariables(networkdm, i, pipes_data[i-eStart]->nx);
	}

	DMNetworkGetVertexRange(networkdm, &vStart, &vEnd);
	for (int i = vStart; i < vEnd; i++) {
		DMNetworkAddComponent(networkdm, i, componentkeys[1], vertices_data[i - vStart]);
		DMNetworkAddNumVariables(networkdm, i, 1);
	}
	
	DMSetUp(networkdm);

	if (rank == 0) {
		PetscFree(connectivity);
		for (int i = 0; i < vertices_count; i++) {
			PetscFree(vertices_data[i]);
		}
		for (int i = 0; i < edges_count; i++) {
			PetscFree(pipes_data[i]);
		}
		PetscFree(vertices_data);
		PetscFree(pipes_data);
	}

	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &n_prcs); CHKERRQ(ierr);
	if (n_prcs > 1) {
		DM distnetworkdm;
		/* Network partitioning and distribution of data */
		ierr = DMNetworkDistribute(networkdm, 0, &distnetworkdm); CHKERRQ(ierr);
		ierr = DMDestroy(&networkdm); CHKERRQ(ierr);
		networkdm = distnetworkdm;
	}

#if SHOW_DIVISION
	PetscInt numComponents;
	DMNetworkComponentGenericDataType *arr;
	PIPEDATA pipe;
	VERTEXDATA vertex;

	ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd); CHKERRQ(ierr);
	ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd); CHKERRQ(ierr);

	ierr = DMNetworkGetComponentDataArray(networkdm, &arr); CHKERRQ(ierr);

	for (int i = eStart; i < eEnd; i++) {
		ierr = DMNetworkGetComponentTypeOffset(networkdm, i, 0, &key, &offset); CHKERRQ(ierr);
		pipe = (PIPEDATA)(arr + offset);
		ierr = DMNetworkGetNumComponents(networkdm, i, &numComponents); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF, "Rank %d ncomps = %d Line %d ---- %d\n", rank, numComponents, pipe->inlet_vertex, pipe->outlet_vertex); CHKERRQ(ierr);
	}

	for (int i = vStart; i < vEnd; i++) {
		ierr = DMNetworkGetComponentTypeOffset(networkdm, i, 0, &key, &offset); CHKERRQ(ierr);
		vertex = (VERTEXDATA)(arr + offset);
		ierr = PetscPrintf(PETSC_COMM_SELF, "Rank %d ncomps = %d Node %d\n", rank, numComponents, vertex->index); CHKERRQ(ierr);
	}
#endif

	ierr = DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd); CHKERRQ(ierr);
	ierr = DMNetworkGetVertexRange(networkdm, &vStart, &vEnd); CHKERRQ(ierr);

	ierr = DMNetworkGetComponentDataArray(networkdm, &components_data); CHKERRQ(ierr);

	PetscMalloc1(eEnd - eStart, &pipes_data);
	PetscMalloc1(vEnd - vStart, &vertices_data);

	for (int i = eStart; i < eEnd; i++) {
		ierr = DMNetworkGetComponentTypeOffset(networkdm, i, 0, &key, &offset); CHKERRQ(ierr);
		pipes_data[i-eStart] = (PIPEDATA)(components_data + offset);
	}

	for (int i = vStart; i < vEnd; i++) {
		ierr = DMNetworkGetComponentTypeOffset(networkdm, i, 0, &key, &offset); CHKERRQ(ierr);
		vertices_data[i-vStart] = (VERTEXDATA)(components_data + offset);
	}

	Vec x;
	DMCreateGlobalVector(networkdm, &x);
	VecSet(x, 2.0);
	VecAssemblyBegin(x);
	VecAssemblyEnd(x);
	VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	Mat J;
	DMCreateMatrix(networkdm, &J);
	MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
	MatView(J, PETSC_VIEWER_STDOUT_WORLD);

	PetscFree(pipes_data);
	PetscFree(vertices_data);

	ierr = DMDestroy(&networkdm);
	ierr = PetscFinalize(); CHKERRQ(ierr);

	return 0;
}