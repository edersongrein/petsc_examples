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
#define __FUNCT__ "main"
int main(int argc, char **argv) {

	PetscInitialize(&argc, &argv, (char*)0, help);

	configure_petsc_options();

	double dt = 0.001;
	double dt_min = 1.0e-3;
	double dt_max = 10.0;
	double final_time = 0.01;

	PetscInt nx = 5;
	PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);

	double temperature_presc[] = { 2.0, 50.0, 55.0, 60.0 };

	double initial_temperature = 10.0;
	double conductivity = 1.0;
	double source_term = 0.0;
	double wall_length = 1.0;

	Params params;
	params.conductivity_ = 1.0;
	params.source_term_ = 0.0;
	params.wall_length_ = 1.0;
	PetscMalloc1(4, &params.temperature_presc_);
	params.temperature_presc_[0] = 2.0;
	params.temperature_presc_[1] = 50.0;
	params.temperature_presc_[2] = 55.0;
	params.temperature_presc_[3] = 60.0;

	TS ts;
	TSCreate(PETSC_COMM_WORLD, &ts);

	//// Single pipe case
	//DM domain;
	//PetscInt dof = 1;
	//PetscInt stencil_width = 1;
	//DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &domain);

	 //Multiple pipes
	DM* pipes;

	DM dm_redundant;
	DMCreate(PETSC_COMM_WORLD, &dm_redundant);
	DMSetType(dm_redundant, DMREDUNDANT);
	RedundantSetSize(dm_redundant, 0, 1);
	DMSetUp(dm_redundant);

	DM domain;
	DMCompositeCreate(PETSC_COMM_WORLD, &domain);

	PetscInt dof = 1;
	PetscInt stencil_width = 1;
	PetscInt npipes = 4;
	for (int i = 0; i < PIPES_SIZE; i++) {
		DM pipe;
		DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &pipe);
		DMCompositeAddDM(domain, pipe);
	}

	DMCompositeAddDM(domain, dm_redundant);

	DMCompositeSetCoupling(domain, FormCoupleLocations);

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
	
	//TSSolve(ts, x);

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

	DMDestroy(&domain);
	TSDestroy(&ts);

	PetscFinalize();

	return 0;
}