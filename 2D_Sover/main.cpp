#include "solver.cpp"

void write_csv(string filename, string col, vector<double> val){

	ofstream myFile(filename);
	
	myFile << col << endl;
	for (int i =0; i < val.size(); ++i)
	{
		myFile << val[i] <<endl;
	}
	myFile.close();
}

int main(int argc, char** argv) {
	double L=100;
	double a=0.01;
	double dt=1;
	
	HeatSolver solver(L, a, 51, dt, 100);

	solver.setSource([](double x, double t) {
    return 5.0;
	});

	// Implicit Solver

	solver.setBoundary(0.0, 0.0);

	solver.setInitial(0.0);

	PetscInitialize(&argc, &argv, NULL, NULL);
	solver.solve_imp();
	PetscFinalize();
	vector<double> imp_solution = solver.getSolution();

	// Explicit Solver

	solver.setBoundary(0.0, 0.0);

	solver.setInitial(0.0);

	solver.solve_exp();
	vector<double> exp_solution = solver.getSolution();

	cout << "Temperature:" << endl;
	for (int i=0; i<imp_solution.size(); i++) {
		cout << "Timp[" << i << "] = " << imp_solution[i] << " Texp[" << i << "] = " << exp_solution[i] << endl;
	}


	
	//write_csv("1DHT.csv", "Temperature", solution);
} 
