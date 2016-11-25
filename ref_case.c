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
	double final_time = 10.0;

	PetscInt nx = 100000;

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

	// Single pipe
	DM domain;
	PetscInt dof = 1;
	PetscInt stencil_width = 1;
	DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &domain);

	// Multiple pipes
	//DM* pipes;
	//pipes = (DM*)malloc(npipes * sizeof(DM));

	//DM dm_redundant;
	//DMCreate(PETSC_COMM_WORLD, &dm_redundant);
	//DMSetType(dm_redundant, DMREDUNDANT);
	//RedundantSetSize(dm_redundant, 0, 1);
	//DMSetUp(dm_redundant);

	//DM domain;
	//DMCompositeCreate(PETSC_COMM_WORLD, &domain);

	//PetscInt dof = 1;
	//PetscInt stencil_width = 1;
	//for (int i = 1; i < npipes; i++) {
	//	DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, nx, dof, stencil_width, NULL, &pipes[i]);
	//	DMCompositeAddDM(domain, pipes[i]);
	//}

	//DMCompositeAddDM(domain, dm_redundant);

	//CompositeSetCoupling(domain);

	TSSetDM(ts, domain);

	Vec F;
	DMCreateGlobalVector(domain, &F);
	TSSetIFunction(ts, F, FormFunctionSinglePipe, &params);

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
	
	TSSolve(ts, x);

	//VecView(x, PETSC_VIEWER_STDOUT_WORLD);

	//for (int i = 0; i < npipes; i++) {
	//	DMDestroy(&pipes[i]);
	//}
	PetscFree(params.temperature_presc_);

	DMDestroy(&domain);
	TSDestroy(&ts);

	PetscFinalize();

	return 0;
}