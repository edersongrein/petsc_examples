#include <stdio.h>

#include "HeatTransfer1Dimpl.h"

static char help[] = "Solves a composite 1D heat transfer problem.\n\n";

void configure_petsc_options() {

	PetscOptionsSetValue(NULL, "-ts_type", "bdf");
	PetscOptionsSetValue(NULL, "-ts_bdf_order", "1");
	PetscOptionsSetValue(NULL, "-ts_adapt_type", "basic");
	PetscOptionsSetValue(NULL, "-ts_bdf_adapt", "");
	PetscOptionsSetValue(NULL, "-ts_monitor", "");

	PetscOptionsSetValue(NULL, "-ts_fd_color", "");
	//PetscOptionsSetValue(NULL, "-mat_view", "draw");
	//PetscOptionsSetValue(NULL, "-draw_pause", "100");
	//PetscOptionsSetValue(NULL, "-is_coloring_view", "");
	//PetscOptionsSetValue(NULL, "-help", "");

}

#undef __FUNCT__
#define __FUNCT__ "CreateGlobalVector"
static PetscErrorCode CreateGlobalVector(DM shell, Vec *x)
{
	PetscErrorCode	ierr;
	DM				*local_dms, composite;
	PetscInt		nDMs, local_size, dm_size, rank;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	ierr = DMShellGetContext(shell, (void**)&composite); CHKERRQ(ierr);
	ierr = DMCompositeGetNumberDM(composite, &nDMs);
	PetscMalloc1(nDMs, &local_dms);
	DMCompositeGetEntriesArray(composite, local_dms);
	local_size = 0;
	for (int p = 0; p < nDMs; p++) {
		DMType dm_type;
		DMGetType(local_dms[p], &dm_type);
		
		PetscBool type_matched = PETSC_FALSE;
		PetscStrcmp(dm_type, DMDA, &type_matched);
		if (type_matched) {
			DMDAGetCorners(local_dms[p], NULL, NULL, NULL, &dm_size, NULL, NULL);
		}

		PetscStrcmp(dm_type, DMREDUNDANT, &type_matched);
		if (type_matched) {
			PetscInt prc_owner;
			DMRedundantGetSize(local_dms[p], &prc_owner, &dm_size);
			if (prc_owner != rank) {
				dm_size = 0;
			}
		}

		local_size += dm_size;
	}

	VecCreate(PETSC_COMM_WORLD, x);
	VecSetSizes(*x, local_size, PETSC_DECIDE);

	ierr = VecSetDM(*x, shell); CHKERRQ(ierr);
	
	ierr = VecSetFromOptions(*x); CHKERRQ(ierr);
	PetscFree(local_dms);

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "CreateMatrix"
static PetscErrorCode CreateMatrix(DM shell, Mat *A)
{
	PetscErrorCode	ierr;
	DM				composite;
	Mat				local_A;
	PetscInt		prc_size;

	ierr = DMShellGetContext(shell, (void**)&composite); CHKERRQ(ierr);
	MPI_Comm_size(PETSC_COMM_WORLD, &prc_size);

	if (prc_size == 1) {
		ierr = DMCreateMatrix(composite, A); CHKERRQ(ierr);
	}
	else{
		ierr = DMCreateMatrix(composite, &local_A); CHKERRQ(ierr);
		MatView(local_A, PETSC_VIEWER_STDOUT_SELF);
		MPI_Barrier(PETSC_COMM_WORLD);

		ierr = MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD, local_A, PETSC_DECIDE, MAT_INITIAL_MATRIX, A); CHKERRQ(ierr);
	}

	//MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
	//MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);

	//MatView(*A, PETSC_VIEWER_STDOUT_WORLD);

	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv) {

	PetscInitialize(&argc, &argv, (char*)0, help);

	configure_petsc_options();

	double dt = 0.001;
	double dt_min = 1.0e-3;
	double dt_max = 10.0;
	double final_time = 100.0;

	PetscInt nx = 2;
	PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);

	int rank, n_prcs;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &n_prcs);

	double initial_temperature = 10.0;
	double conductivity = 1.0;
	double source_term = 0.0;
	double wall_length = 1.0;

	PetscInt dof = 1;
	PetscInt stencil_width = 1;

	Params params;
	params.conductivity_ = 1.0;
	params.source_term_ = 0.0;
	params.wall_length_ = 1.0;
	PetscMalloc1(4, &params.temperature_presc_);
	params.temperature_presc_[0] = 2.0;
	params.temperature_presc_[1] = 10.0;
	params.temperature_presc_[2] = 55.0;
	params.temperature_presc_[3] = 60.0;

	TS ts;
	TSCreate(PETSC_COMM_WORLD, &ts);

	////////////////////////////////////////////////////////////
	////				Single pipe							////
	////////////////////////////////////////////////////////////
	//DM domain;
	//DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &domain);

	//////////////////////////////////////////////////////////////
	//////				SHELL DM 							////
	//////////////////////////////////////////////////////////////

	//PetscInt prc_owner[PIPES_SIZE];
	//int owner = 0;
	//for (int i = 0; i < PIPES_SIZE; i++) {
	//	if (owner >= n_prcs) {
	//		owner = 0;
	//	}
	//	prc_owner[i] = owner++;
	//}

	//DM subdomain;
	//DMCompositeCreate(PETSC_COMM_SELF, &subdomain);

	//for (int p = 0; p < PIPES_SIZE; p++) {
	//	if (prc_owner[p] == rank) {
	//		DM pipe;
	//		DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &pipe);
	//		DMCompositeAddDM(subdomain, pipe);
	//	}
	//}

	//DM dm_redundant;
	//DMCreate(PETSC_COMM_WORLD, &dm_redundant);
	//DMSetType(dm_redundant, DMREDUNDANT);
	//RedundantSetSize(dm_redundant, n_prcs - 1, 1);
	//DMSetUp(dm_redundant);

	//DMCompositeAddDM(subdomain, dm_redundant);
	////DMCompositeSetCoupling(subdomain, FormCoupleLocations);

	//DM domain;
	//DMShellCreate(PETSC_COMM_WORLD, &domain);

	//DMShellSetContext(domain, subdomain); 
	//DMShellSetCreateGlobalVector(domain, CreateGlobalVector);
	//DMShellSetCreateMatrix(domain, CreateMatrix);

	//////////////////////////////////////////////////////////////
	//////				Multiple pipes						////
	//////////////////////////////////////////////////////////////
	//DM* pipes;

	//DM dm_redundant;
	//DMCreate(PETSC_COMM_WORLD, &dm_redundant);
	//DMSetType(dm_redundant, DMREDUNDANT);
	//RedundantSetSize(dm_redundant, 0, 1);
	//DMSetUp(dm_redundant);

	//DM domain;
	//DMCompositeCreate(PETSC_COMM_WORLD, &domain);

	//for (int i = 0; i < PIPES_SIZE; i++) {
	//	DM pipe;
	//	DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &pipe);
	//	DMCompositeAddDM(domain, pipe);
	//}

	//DMCompositeAddDM(domain, dm_redundant);

	//DMCompositeSetCoupling(domain, FormCoupleLocations);

	//////////////////////////////////////////////////////////////

	TSSetDM(ts, domain);

	Vec F;
	DMCreateGlobalVector(domain, &F);

	//TSSetIFunction(ts, F, FormFunctionSinglePipe, &params);
	TSSetIFunction(ts, F, FormFunction, &params);

	Vec x;
	DMCreateGlobalVector(domain, &x);
	VecSet(x, initial_temperature);
	
	TSSetDuration(ts, PETSC_DEFAULT, final_time);
	TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);

	TSSetInitialTimeStep(ts, 0.0, dt);

	TSSetProblemType(ts, TS_NONLINEAR);

	TSAdapt adapt;
	TSGetAdapt(ts, &adapt);
	TSAdaptSetStepLimits(adapt, dt_min, dt_max);
	
	TSSetFromOptions(ts);
	
	VecSet(F, 1.0);
	VecAssemblyBegin(F);
	VecAssemblyEnd(F);

	VecView(F, PETSC_VIEWER_STDOUT_WORLD);
	TSSolve(ts, x);

	//VecView(x, PETSC_VIEWER_STDOUT_SELF);
	//SNES snes;
	//TSGetSNES(ts, &snes);
	//SNESConvergedReason reason;
	//SNESGetConvergedReason(snes, &reason);
	//PetscPrintf(PETSC_COMM_WORLD, "snes reason %i", reason);

	//Vec analytical;
	//DMCreateGlobalVector(domain, &analytical);

	//PetscInt ix, mx;
	//DMDAGetCorners(domain, &ix, NULL, NULL, &mx, NULL, NULL);

	//PetscScalar *analytical_array;
	//DMDAVecGetArray(domain, analytical, &analytical_array);

	//double dT_dx = (params.temperature_presc_[1] - params.temperature_presc_[0]) / wall_length;
	//for (int i = ix; i < ix + mx; i++) {
	//	double x = i * wall_length / (nx - 1);
	//	double T = params.temperature_presc_[0] + dT_dx * x;
	//	analytical_array[i] = T;
	//}

	//DMDAVecRestoreArray(domain, analytical, &analytical_array);

	//VecAssemblyBegin(analytical);
	//VecAssemblyEnd(analytical);

	//Vec error;
	//DMCreateGlobalVector(domain, &error);

	//VecWAXPY(error, -1.0, x, analytical);

	//PetscReal error_norm;
	//VecNorm(error, NORM_2, &error_norm);

	//PetscPrintf(PETSC_COMM_WORLD, "\nError %g", error_norm);

	PetscFree(params.temperature_presc_);

	DMDestroy(&subdomain);
	//DMDestroy(&domain);
	TSDestroy(&ts);

	PetscFinalize();

	return 0;
}