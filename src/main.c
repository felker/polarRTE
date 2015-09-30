#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "visit_writer.h"

//select solver options
#define X2_PERIODIC 1 //flag to wrap phi coordinate and automatically mesh entire circle
#define AUTO_TIMESTEP 0 //flag for automatically setting dt such that max{cfl_array} = CFL
#define CFL 0.8 //if set to 1.0, 1D constant advection along grid is exact. 
#define OUTPUT_INTERVAL 0 //how many timesteps to dump simulation data. 0 for only last step, 1 for every step

#define SECOND_ORDER //flag to turn on van Leer flux limiting

#undef NEW_FLUX //flag to use next incarnation of flux formula
#undef ANALYTIC_SOLUTION //skip timestepping and output steady state solution in Cartesian basis
//bounds of outer radial boundary conditions for searchlight (inclusive)
//SPATIAL POSITIONS

#undef TEST1 //original crossing beam test over continuous range of cartesian angles
#undef TEST2 //isotropic, thick point source
#define TEST3 //isotropic, single point source
#undef TEST4 //beam at delta(\xi), avoiding hole

#define PHI_I M_PI/6
#define PHI_F M_PI/3 /*
#define PHI_I 3*M_PI/4
#define PHI_F 5*M_PI/6 
		     */

//MOMENTUM DIRECTIONS IN LOCAL COORDINATE BASIS
//NOT USED!
#define XA1_I M_PI
#define XA1_F 3*M_PI/2 //7*M_PI/6 

//momentum directions in global cartesian basis

//it makes sense to define the BC in terms of discrete cartesian bins because they can be represented perfectly as a discrete approximation to a continuous angular progile.
//however, this assumes a continuous range of angles (XA1_I_C != XA1_F_C). Sample angle for a bin must fall in this range. In the future, we should do the overlap of the solid angle range with the boundary condition

//they may map to continous polar angular bins
//the same is true vise versa: discrete polar bins map to continuous cartesian angular bins
#define XA1_I_C 4*M_PI/3
#define XA1_F_C 3*M_PI/2 //7*M_PI/6

#undef DELTA_ANGLE_C 0 //turn flag on if you want to use a single cartesian direction 

#undef TRASH

double stream_function(double x, double y);
double X_physical(double, double);
double Y_physical(double, double);
void vector_coordinate_to_physical(double vr, double vphi, double vphase, double phi, double r, double *vx, double *vy, double *vz);
void vector_physical_to_coordinate(double vx, double vy, double phi, double r, double *vr, double *vphi);
double xA_coordinate_to_physical(double xa1, double x2);
void velocity_physical(double x_b,double y_b,double *vx,double *vy);
void velocity_coordinate(double r_b,double phi_b,double *vr,double *vphi);
double initial_condition(double x, double y, double xa1);
double bc_x1i(double x, double y, double xa1, double t);
double bc_x1f(double x, double y, double xa1, double t);
double bc_x2i(double x, double y, double xa1, double t);
double bc_x2f(double x, double y, double xa1, double t);
double flux_PLM(double ds,double *imu);
float find_max(float a[], int n);
float find_min(float a[], int n); 
float sum(float a[], int n);
double gaussian(double x_0, double y_0,double x,double y);
int uniform_angles2D(int N, double phi_z, double *pw, double **mu, double **mu_b, double *xa1,double *xa1_b);
double **allocate_2D_contiguous(double *a, int n1, int n2);
double ***allocate_3D_contiguous(double *a, int n1, int n2, int n3);
double find_max_double(double a[], int n);
float bc_x1f_polar(double phi, double xa1, double t);
double flux_PLM_athena(double r[3], int dir, double dt, double ds, double vel, double imu[3]);
double bc_interior(double r, double phi, double xa1, double t);
void gaussianelim(double **A, double *b, double *x, int n, int pivot);
int permutation(int i, int j, int k, int ** pl, int np);
int bruls_angles2D(int N, double phi_z, double *pw, double **mu, double **mu_b, double *xa1,double *xa1_b);
int cmpfunc (const void * a, const void * b);

#ifdef DELTA_ANGLE_C
float bc_x1f_polar_delta(double phi, double xa1_i, double xa1_f, double dxa1, double t);
#endif

double f(double phi, double x, double y, double r_max,double angle_c);
double df(double phi, double x, double y, double r_max);
float analytic_solution(double x,double y, double r_max,double angle_c);
double newton_raphson(double x, double y, double r_max, double angle_c, double x0, double allerr, int maxmitr);

int main(int argc, char **argv){
  int i,j,k,l,n; 
  int next, next2, prev, prev2; //for indexing third dimension
  int nsteps=500;
  double dt =0.02;
  
  /* Computational (2D polar) grid coordinates */
  int nx1 = 50;
  int nx2 = 50; 

  /* Angular parameter */
  int xa1_uniform = 1; 
  int nxa1;
  double *dxa1;  //angular width of cell 
  int N_bruls; //analogous to N in MATLAB code. must be even <=12
  if (xa1_uniform){
    nxa1 = 8;
  } 
  else {
    N_bruls = 4;
    nxa1 = N_bruls*(N_bruls+2)/2;
  }
  dxa1 = malloc(sizeof(double)*nxa1);

  double phi_z = M_PI/2; 

  //number of ghost cells on both sides of each dimension
  //only need 1 for piecewise constant method
  //need 2 for piecewise linear reconstruction
  int num_ghost = 2;
  int nx1_r = nx1;
  int nx2_r = nx2;
  nx1 += 2*num_ghost; 
  nx2 += 2*num_ghost; 

  /*non-ghost indices */
  int is = num_ghost;
  int ie= is+nx1_r; 
  int js = num_ghost;
  int je = js+nx2_r;
  int ks = 0;
  int ke = nxa1;

  //convention: these ranges refer to the non-ghost cells
  //however, the ghost cells have real coordinate interpretations
  //this means that we must be sure that the number of ghost cells makes sense with the range of coordinates
  //this is a big problem if the phi polar coordinate runs the whole range [0,2pi) for example

  //further, all mesh structures (zonal and nodal) will be nx1 x nx2, although we may not fill the ghost entries with anything meaningful
  //this is to standardize the loop indexing from is:ie

  /*another convention: when the phi coordinate is periodic, do we repeat the boundary mesh points? yes for now */

  double lx1 = 2.0;//these values are inclusive [x1_i, x1_f]
  double lx2 = M_PI;
  double x1_i = 0.5;
  double x2_i = 0.0;

  double dx2 = lx2/(nx2_r-1);   
  double x2_f = x2_i + lx2;

  if(X2_PERIODIC){
    dx2 = 2*M_PI/(nx2_r);   
    lx2 = dx2*(nx2_r-1); 
    x2_i = 0.0;
  }

  double dx1 = lx1/(nx1_r-1);   
  double x1_f = x1_i + lx1;

  printf("dx1=%lf dx2=%lf \n",dx1,dx2); 
  
  /*Cell centered (zonal) values of computational coordinate position */
  double *x1 = (double *) malloc(sizeof(double)*nx1); 
  double *x2 = (double *) malloc(sizeof(double)*nx2);
  x1[is] = x1_i;
  x2[js] = x2_i;
  for(i=is+1; i<ie; i++){ 
    x1[i] = x1[i-1] + dx1;
    //    printf("%lf\n",x1[i]);
  }
  for(i=js+1; i<je; i++){
    x2[i] = x2[i-1] + dx2;
    //printf("%lf\n",x2[i]);
  } 

  if(X2_PERIODIC){//duplicate the phis at the edges of circle for easy peridoicity 
    x2[js-1] = x2[je-1];
    x2[je] = x2[js];
  }
    
  /*Mesh edge (nodal) values of computational coordinate position */
  double *x1_b = (double *) malloc(sizeof(double)*(nx1+1)); 
  double *x2_b = (double *) malloc(sizeof(double)*(nx2+1));
  x1_b[is] = x1_i - dx1/2;
  x2_b[js] = x2_i - dx2/2;
  for(i=is+1; i<=ie; i++){ 
    x1_b[i] = x1_b[i-1] + dx1;
    // printf("x1_b[%d] = %lf\n",i,x1_b[i]);
  }
  for(i=js+1; i<=je; i++){
    x2_b[i] = x2_b[i-1] + dx2;
    //    printf("x2_b[%d] = %lf\n",i,x2_b[i]);
  } 

  /*Cell centered (zonal) values of physical coordinate position */
  //These must be 2D arrays since coordinate transformation is not diagonal
  //indexed by x1 columns and x2 rows.

  /*CONVENTION: flip the dimension ordering in all multiD arrays due to row major ordering of C */
  double *dataX = (double *) malloc(sizeof(double)*nx1*nx2);
  double *dataY = (double *) malloc(sizeof(double)*nx1*nx2);
  double **x = allocate_2D_contiguous(dataX,nx2,nx1); 
  double **y = allocate_2D_contiguous(dataY,nx2,nx1); 

  /*precompute coordinate mappings */
  for(j=js; j<je; j++){
    for(i=is; i<ie; i++){
      x[j][i] = X_physical(x1[i],x2[j]); 
      y[j][i] = Y_physical(x1[i],x2[j]); 
      //      printf("x,y=%lf,%lf\n",x[j][i],y[j][i]);
    }
  }

  /*Mesh edge (nodal) values of physical coordinate position */
  double *dataXb = (double *) malloc(sizeof(double)*(nx1+1)*(nx2+1));
  double *dataYb = (double *) malloc(sizeof(double)*(nx1+1)*(nx2+1));
  double **x_b = allocate_2D_contiguous(dataXb,nx2+1,nx1+1); 
  double **y_b = allocate_2D_contiguous(dataYb,nx2+1,nx1+1); 
  for(j=js; j<=je; j++){
    for(i=is; i<=ie; i++){
      x_b[j][i] = X_physical(x1_b[i],x2_b[j]); 
      y_b[j][i] = Y_physical(x1_b[i],x2_b[j]); 
    }
  }
  
  /*Discretize photon directions */
  double *xa1 = (double *) malloc(sizeof(double)*nxa1); 
  double *xa1_b = (double *) malloc(sizeof(double)*(nxa1+1)); 
  double *datamu = (double *) malloc(sizeof(double)*nxa1*3); 
  double *datamu_b = (double *) malloc(sizeof(double )*3*(nxa1+1));//identify with left and bottom boundaries 
  double *pw = (double *) malloc(sizeof(double)*nxa1); 
  double **mu = allocate_2D_contiguous(datamu,nxa1,3); //CONVENTION: xa1 is the first dimension, component is second dim 
  double **mu_b = allocate_2D_contiguous(datamu_b,nxa1+1,3); 

  if (xa1_uniform){
    nxa1 = uniform_angles2D(nxa1, phi_z, pw, mu, mu_b, xa1,xa1_b); 
  }
  else{
    bruls_angles2D(N_bruls, phi_z, pw, mu, mu_b, xa1,xa1_b); 
  }

  for (i=0; i<nxa1-1; i++)
    dxa1[i] = xa1_b[i+1] - xa1_b[i];
  //last boundary should be 2pi 
  dxa1[nxa1-1] = 2*M_PI - xa1_b[nxa1-1];
  
  /*Coordinate cell capacity */
  double *datakappa= (double *) malloc(sizeof(double)*nx1*nx2*nxa1);
  double ***kappa = allocate_3D_contiguous(datakappa,nxa1+1,nx2,nx1); 

  //cell centered velocities require the ability to access is-1,js-1
  //  x1[is-1] = x1[is] - dx1; //only works if dx1 < 0.5
  
  // switch to phase volume formalism:
  double ***vol = allocate_3D_contiguous(datakappa,nxa1+1,nx2,nx1); 

  for(k=ks; k<=ke; k++){
    for(j=js-1; j<=je; j++){
      for(i=is-1; i<=ie; i++){
	if (i==ie)
	  kappa[k][j][i] = x1[ie-1];
	else if (i==(is-1))
	  kappa[k][j][i] = x1[is];
	else
	  kappa[k][j][i] = x1[i];
	//remove all references to angular dimension for now:
	vol[k][j][i] = dx1*dx2; 
	if (i==ie)
	  vol[k][j][i] *= x1[ie-1];
	else if (i==(is-1))
	  vol[k][j][i] *= x1[is];
	else 
	  vol[k][j][i] *= x1[i];

	if (k==ke)
	  vol[k][j][i] *= dxa1[ks];
	else
	  vol[k][j][i] *= dxa1[k];
	//	printf("vol =%lf dxa1[%d]= %lf\n" ,vol[k][j][i],k,dxa1[k]); 
      }
    }
  } 


  //uniform 
  //2*M_PI/(nxa1); //dont mesh all the way to 2pi
  
  for(k=ks; k<ke; k++)
    printf("xa1[%d] = %lf \n",k,xa1[k]);
  for(k=ks; k<=ke; k++)
    printf("xa1_b[%d] = %lf \n",k,xa1_b[k]);   
  for(k=ks; k<ke; k++)
    printf("dxa1[%d] = %lf \n",k,dxa1[k]);   
  for(k=ks; k<ke; k++)
    printf("mu[%d] = (%lf,%lf) \n",k,mu[k][0],mu[k][1]);
  for(k=ks; k<=ke; k++)
    printf("mu_b[%d] = (%lf,%lf) \n",k,mu_b[k][0],mu_b[k][1]);   
  
  //Check Bruls Quadrature conditions-- only makes sense in 3D propagation
  return(0);
  /*Average normal edge velocities */
  //now we move from cell centered quantities to edge quantities 
  //the convention in this code is that index i refers to i-1/2 edge
  double *dataU = malloc(sizeof(double)*nxa1*nx2*nx1);
  double *dataV = malloc(sizeof(double)*nxa1*nx2*nx1);
  double *dataW = malloc(sizeof(double)*nxa1*nx2*nx1);

  double ux,vy,wz,temp; 
  double ***U = allocate_3D_contiguous(dataU,nxa1+1,nx2,nx1);
  double ***V = allocate_3D_contiguous(dataV,nxa1+1,nx2,nx1);
  double ***W = allocate_3D_contiguous(dataW,nxa1+1,nx2,nx1);
  double ***source = allocate_3D_contiguous(dataW,nxa1+1,nx2,nx1);//not using contiguous array right now. debug

  //Need to be able to reference ghost cells, for which x1[] is undefined 
  for(k=ks; k<=ke; k++){    //this incorrectly handles dxa1[ke] which is undefined
    for(j=js-1; j<=je; j++){       
      for(i=is-1; i<=ie; i++){
	if (k==ke)
	  next = ks; 
	else 
	  next = k+1; 
	//Midpoint approximations to edge velocities 
	/*	U[k][j][i] = mu[k][0];
	V[k][j][i] = mu[k][1];
	W[k][j][i] = -mu[k][1]/kappa[k][j][i]; */
	//Compute exact average edge velocities 
	U[k][j][i] = (sin(xa1_b[next]) - sin(xa1_b[k]))/dxa1[k]; 
	V[k][j][i] = (-cos(xa1_b[next]) + cos(xa1_b[k]))/dxa1[k]; 
	W[k][j][i] = -sin(xa1_b[k]);
	if (i == ie)
	  W[k][j][i] /= x1[ie-1]; 
	else if (i == is-1)
	  W[k][j][i] /= x1[is];
	else
	  W[k][j][i] /= x1[i]; 
	source[k][j][i] = 0.0;
      }
    }
  }  
  //ERROR: need to pass absolute values to max finder
  //  printf("max|U| = %lf max|V| = %lf max|W| = %lf\n",find_max_double(dataU,nxa1*nx1*nx2),find_max_double(dataV,nxa1*nx1*nx2),find_max_double(dataW,nxa1*nx1*nx2));

  /*Check CFL condition, reset timestep */
  double *datacfl = malloc(sizeof(double)*nxa1*nx2*nx1); 
  double ***cfl_array =  allocate_3D_contiguous(datacfl,nxa1,nx2,nx1);
  
  //debug:
  int index=0; 
  int indexJ;   /* for computing zeroth angular moment of the radiation field*/

  for(k=0; k<nxa1; k++){ //based on edge velocities or cell centered u,v?
    for(j=0; j<nx2; j++){
      for(i=0; i<nx1; i++){
	if (i >=is && i< ie && j >=js && j <je)
	  cfl_array[k][j][i] = fabs(U[k][j][i])*dt/dx1 + fabs(V[k][j][i])*dt/(x1_b[i]*dx2) + fabs(W[k][j][i])*dt/(x1_b[i]*dxa1[k]); //use boundary radius
	  //	  printf("i=%d CFL = %lf U =%lf\n",i,cfl_array[k][j][i],U[k][j][i]);	
	else
	  cfl_array[k][j][i] =0.0; 
	datacfl[index] = cfl_array[k][j][i];
	index++;
      }
    }
  }

  //find maximum CFL value in domain
  double  max_cfl = find_max_double(datacfl,nxa1*nx1*nx2); 
  index =0;
  printf("Largest CFL number = %lf\n",max_cfl); 
  if (max_cfl > CFL || AUTO_TIMESTEP){//reset timestep if needed
    if (max_cfl ==0) //dont divide by 0
      exit(1);
    dt = CFL*dt/max_cfl; 
    for(k=0; k<nxa1; k++){ //ERROR IN ADVECTION. ARRAY STARTS AT 1
      for(j=0; j<nx2; j++){
	for(i=0; i<nx1; i++){ 
	if (i >=is && i< ie && j >=js && j <je)
	  cfl_array[k][j][i] = fabs(U[k][j][i])*dt/dx1 + fabs(V[k][j][i])*dt/(x1_b[i]*dx2) + fabs(W[k][j][i])*dt/(x1_b[i]*dxa1[k]); //use boundary radius
	else
	  cfl_array[k][j][i] =0.0; 
	}
	datacfl[index] = cfl_array[k][j][i];
	index++;
      }
    } 
  }
  //debug
  max_cfl = find_max_double(datacfl,nxa1*nx1*nx2);
  printf("Largest CFL number = %lf\n",max_cfl); 

  /*Conserved variable on the computational coordinate mesh*/
  double *dataI =  malloc(sizeof(double )*nx1*nx2*nxa1);
  double ***I = allocate_3D_contiguous(dataI,nxa1,nx2,nx1); 

  /*Initial condition */
  //specified as a function of cartesian physical cooridnates, as is the stream function
  for(k=ks; k<ke; k++){
    for(j=js; j<je; j++){
      for(i=is; i<ie; i++){
	I[k][j][i] = initial_condition(x[j][i],y[j][i],xa1[k]); 
      }
    }
  }

  /* Net fluxes in each dimension at each timestep */
  double U_plus,U_minus,V_plus,V_minus,W_plus,W_minus;
  double *dataFlux = malloc(sizeof(double)*nx1*nx2*nxa1);
  double ***net_flux = allocate_3D_contiguous(dataFlux,nxa1,nx2,nx1); 

  /* Using Visit VTK writer */
  char filename[20];
  /*CONVENTION: simply make angular dimension the third spatial dimension for now. Later may want to create separate variables */
  int dims[] = {nx1_r+1, nx2_r+1, nxa1+1}; //dont output ghost cells. //nodal variables have extra edge point
  int nvars = 3;
  int vardims[] = {1, 1, 3}; //I is a scalar, J is a scalar, velocity is a 3-vector 
  int centering[] = {0, 0, 1}; // I,J are cell centered, velocity is defined at edges
  const char *varnames[] = {"I","J", "edge_velocity"};
  /* Curvilinear mesh points stored x0,y0,z0,x1,y1,z1,...*/
  //An array of size nI*nJ*nK*3 . These points are nodal, not zonal
  float *pts = (float *) malloc(sizeof(float)*(nx1_r+1)*(nx2_r+1)*(nxa1+1)*3); //check angular dimension size
  //The array should be layed out as (pt(i=0,j=0,k=0), pt(i=1,j=0,k=0), ...
  //pt(i=nI-1,j=0,k=0), pt(i=0,j=1,k=0), ...).
  index=0; 
  for(k=ks; k<=ke; k++){
    for(j=js; j<=je; j++){
      for(i=is; i<=ie; i++){
	pts[index] = x_b[j][i];
	pts[++index] = y_b[j][i];
	pts[++index] = xa1_b[k]; 
	index++;
      }
    }
  }
  
  /* pack U,V,W into a vector */
  float *edge_vel = (float *) malloc(sizeof(float)*(nx1_r+1)*(nx2_r+1)*(nxa1+1)*3); //An array of size nI*nJ*nK*3 
  index=0; 
  for(k=ks; k<=ke; k++){
    for(j=js; j<=je; j++){ //ERROR: j=je ghost cells are messed up since U,V,W arent initialized that far
      for(i=is; i<=ie; i++){

	//	if (j==je)
	//vector_coordinate_to_physical(U[k][j][i],V[k][j][i],W[k][j][i],x2_b[0],&ux,&vy,&wz);
	//else
	vector_coordinate_to_physical(U[k][j][i],V[k][j][i],W[k][j][i],x2_b[j],x1_b[i],&ux,&vy,&wz);
	//	printf("k,j,i =%d,%d,%d \n",k,j,i);
        //printf("U,V,W = %lf,%lf,%lf\n",U[k][j][i],V[k][j][i],W[k][j][i]);/*
	/*edge_vel[index] = U[k][j][i];
	edge_vel[++index] = V[k][j][i]; 
	edge_vel[++index] = W[k][j][i]; */
	edge_vel[index] = ux;
	edge_vel[++index] = vy;
	edge_vel[++index] = wz;
	index++;
      }
    }
  } 

  //  vars       An array of variables.  The size of vars should be nvars.
  //                 The size of vars[i] should be npts*vardim[i].
  float *realI, *realJ; //excludes spatial ghost cells
  realI =(float *) malloc(sizeof(float)*nx1_r*nx2_r*nxa1);//ERROR IN ADVECTION. SIZEOF(DOUBLE)
  realJ =(float *) malloc(sizeof(float)*nx1_r*nx2_r*nxa1); //unfortunately, if it lives on the same 3D mesh, must be duplicated at each z/nxa1 height
  float *vars[] = {(float *) realI,(float *) realJ, (float *)edge_vel};
  /*  double maxI[12];
  for (k=ks; k <ke; k++){ //debug individual angular bins
    maxI[k] = find_max(realI+(k*nx1_r*nx2_r),nx1_r*nx2_r);
    printf("%d angular bin theta %lf, max{I} = %lf \n",k,xa1[k],maxI[k]); 
    } */


#ifdef ANALYTIC_SOLUTION
  index=0;
  //  int max_iter =500;
  //double err_tol = 1e-4;
  float *realI_cartesian; //excludes spatial ghost cells
  realI_cartesian =(float *) malloc(sizeof(float)*nx1_r*nx2_r*nxa1);

  double x_source, y_source, slope; 
  double xa1_source; //polar angular bin from which the hypothetical beam should come from
  double phi_source; //polar spatial bin from which the hypothetical beam should come from
  double angle_c; 
  for (k=ks; k<ke; k++){
    for (j=js; j<je; j++){
      for (i=is; i<ie; i++){ //if the center position is in the path?
	//	phi_source = newton_raphson(x[j][i], y[j][i], x1_b[ie],xa1[k], M_PI, err_tol,max_iter);
	//	printf("x,y = %lf,%lf theta = %lf \n",x[j][i],y[j][i],xa1[k]);
	angle_c = xa1[k]; 
	//	propagation is restricted to z/angular planes in cartesian
#if defined(DELTA_ANGLE_C)
	if (fmod(fabs(DELTA_ANGLE_C - xa1_b[k]),2*M_PI) <= dxa1 &&  fmod(fabs(DELTA_ANGLE_C - xa1_b[k+1]),2*M_PI) <= dxa1){
	  //	if (DELTA_ANGLE_C >= xa1_b[k] && DELTA_ANGLE_C <= xa1_b[k+1]){
          slope = tan(DELTA_ANGLE_C);
          if ((y[j][i] <= slope*(x[j][i] - x1_b[ie]*cos(PHI_F)) + x1_b[ie]*sin(PHI_F)) &&
              (y[j][i] >= slope*(x[j][i] - x1_b[ie]*cos(PHI_I)) + x1_b[ie]*sin(PHI_I))){
            realI_cartesian[index] = 1.0; 	  printf("located!\n");
	}
        }
        else{
          realI_cartesian[index] = 0.0;
        }
#else
	if(angle_c >=XA1_I_C && angle_c <= XA1_F_C){
	  slope = tan(angle_c);
	  if ((y[j][i] <= slope*(x[j][i] - x1_b[ie]*cos(PHI_F)) + x1_b[ie]*sin(PHI_F)) &&
	      (y[j][i] >= slope*(x[j][i] - x1_b[ie]*cos(PHI_I)) + x1_b[ie]*sin(PHI_I)))
	    realI_cartesian[index] = 1.0; 
	}
	else{
	  realI_cartesian[index] = 0.0; 
	}
#endif
	index++; 
	
 
#ifdef TRASH
	if (fmod(angle_c, M_PI/2) !=0.0){
	  slope = tan(angle_c); 
	  if (angle_c > M_PI/2 && angle_c < 3*M_PI/2){ // rays coming from right hemisphere,more positive x
	    // take larger root
	    x_source = (-2*(slope*y[j][i] -2*x[j][i]*slope*slope) + sqrt(4*pow((slope*y[j][i] -2*x[j][i]*slope*slope),2) - 4*(slope*slope +1)*(slope*slope*x[j][i]*x[j][i] - 2*slope*y[j][i]*x[j][i] + y[j][i]*y[j][i] - pow(x1_b[ie],2))))/(2*slope*slope +2); 
	    //	    y_source = slope*(x_source - x[j][i]) + y[j][i]; 
	    y_source = sqrt(x1_b[ie]*x1_b[ie] - x_source*x_source); 
	  }
	  else{ // rays coming from left hemisphere, smaller x //the root should always be real
	    // take smaller root
	    x_source = (-2*(slope*y[j][i] -2*x[j][i]*slope*slope) - sqrt(4*pow((slope*y[j][i] -2*x[j][i]*slope*slope),2) - 4*(slope*slope +1)*(slope*slope*x[j][i]*x[j][i] - 2*slope*y[j][i]*x[j][i] + y[j][i]*y[j][i] - pow(x1_b[ie],2))))/(2*slope*slope +2); 
	    //	    y_source = slope*(x_source - x[j][i]) + y[j][i]; 
	    y_source = sqrt(x1_b[ie]*x1_b[ie] - x_source*x_source); 
	  }
	  if ((y_source*y_source + x_source*x_source) != x1_b[ie]*x1_b[ie]){
	    printf("source point is not on the circle!\n");
	    printf("x,y = %lf,%lf theta = %lf rmax = %lf\n",x[j][i],y[j][i],xa1[k],x1_b[ie]);
	    printf("x_s,y_s = %lf,%lf slope =%lf\n",x_source,y_source,slope);
	    printf("phi = %lf xa1_polar = %lf I = %lf\n",phi_source,xa1_source,realI[index]);
	    printf("root1 = %lf root2 = %lf\n",(-2*(slope*y[j][i] -2*x[j][i]*slope*slope) + sqrt(4*pow((slope*y[j][i] -2*x[j][i]*slope*slope),2) - 4*(slope*slope +1)*(slope*slope*x[j][i]*x[j][i] - 2*slope*y[j][i]*x[j][i] + y[j][i]*y[j][i] - pow(x1_b[ie],2))))/(2*slope*slope +2),(-2*(slope*y[j][i] -2*x[j][i]*slope*slope) -sqrt(4*pow((slope*y[j][i] -2*x[j][i]*slope*slope),2) - 4*(slope*slope +1)*(slope*slope*x[j][i]*x[j][i] - 2*slope*y[j][i]*x[j][i] + y[j][i]*y[j][i] - pow(x1_b[ie],2))))/(2*slope*slope +2));
	    return(1);
	  
	  }
	}
	else if (angle_c == 0.0 || angle_c == 2*M_PI) { //horizontal from the left
	  y_source = y[j][i]; 
	  x_source = -sqrt(pow(x1_b[ie],2) - pow(y_source,2));
	}
	else if (angle_c == M_PI) { //horizontal from the right
	  y_source = y[j][i]; 
	  x_source = +sqrt(pow(x1_b[ie],2) - pow(y_source,2));
	}
	else if (angle_c == M_PI/2){ //vertical line from the bottom
	  x_source = x[j][i];
	  y_source = -sqrt(pow(x1_b[ie],2) - pow(x_source,2));
	}
	else if (angle_c == 3*M_PI/2){ //vertical line from the top
	  x_source = x[j][i];
	  y_source = +sqrt(pow(x1_b[ie],2) - pow(x_source,2));
	}
	phi_source = atan2(y_source,x_source) + M_PI;  
	xa1_source = fmod(angle_c - phi_source+2*M_PI,2*M_PI); //angle_c is [0,2pi). phi_source is [0,2pi); 
	if (phi_source > PHI_I && phi_source <= PHI_F && xa1_source > XA1_I && xa1_source <= XA1_F)
	  realI[index] = 1.0; 
	else 
	  realI[index] = 0.0; 
	printf("x,y = %lf,%lf theta = %lf rmax = %lf\n",x[j][i],y[j][i],xa1[k],x1_b[ie]);
	printf("x_s,y_s = %lf,%lf slope =%lf\n",x_source,y_source,slope);
	printf("phi = %lf xa1_polar = %lf I = %lf\n",phi_source,xa1_source,realI[index]);
	index++;
#endif TRASH
	//	if (phi_source != phi_source){ // nan
	//
	// return(1);
	//}
	//	realI[index] = analytic_solution(x[j][i],y[j][i],x1_b[ie],xa1[k] + x2[j]);
      }
    }
  }
  /*Transform analytic solution from cartesian basis to polar basis */
  /*one issue is that discrete angular representation of cartesian solution maps to continuous polar angular representation */
  //just place all intensity in corresponding bin.

  for (k=0; k<nxa1; k++){ //loop over cartesian angular bins
    for (j=0; j<nx2_r; j++){
      for (i=0; i<nx1_r; i++){ 
	phi_source = x2[j+num_ghost]; 
	xa1_source = fmod(xa1[k]- phi_source+2*M_PI,2*M_PI); //find corresponding polar angular bin
	for (l=ks; l<ke; l++){
	  if (xa1_source >= xa1_b[l] && xa1_source <= xa1_b[l+1]){
	    realI[nx1_r*nx2_r*l + nx1_r*j + i] += realI_cartesian[nx1_r*nx2_r*k + nx1_r*j + i];}
	}
      }
    }
  }
  /* Compute zeroth moment of radiation field */
  //zero out J array:
  indexJ=0;
  for (j=0; j<nx2_r; j++){
    for (i=0; i<nx1_r; i++){
      realJ[indexJ] = 0;
      indexJ++;
    }
  }
  index = 0; 
  for (k=ks; k<ke; k++){
    indexJ=0; 
    for (j=js; j<je; j++){
      for (i=is; i<ie; i++){
	realJ[indexJ] += (float) realI[index]*pw[k];
	indexJ++;
	index++;
      }
    }
  }
  /* Copy the radiation energy field from first angular bin to all angular bins */
  indexJ=nx1_r*nx2_r; 
  for (k=ks+1; k<ke; k++){
    for (j=0; j<nx2_r; j++){
      for (i=0; i<nx1_r; i++){
	realJ[indexJ] = realJ[j*nx1_r+i]; 
	indexJ++;
      }
    }
  }
  /* uncomment next line for outputting analytic solution in cartesian basis */
  //  vars[0] = (float *) realI_cartesian;

  sprintf(filename,"analytic-rte.vtk"); 
  write_curvilinear_mesh(filename,1,dims, pts, nvars,vardims, centering, varnames, vars);
  return(0);
#endif
  /* Output initial condition */
  index=0; 
  //zero out J array:
  indexJ=0;
  for (j=0; j<nx2_r; j++){
    for (i=0; i<nx1_r; i++){
      realJ[indexJ] = 0;
      indexJ++;
    }
  }
  for (k=ks; k<ke; k++){
    indexJ=0; 
    for (j=js; j<je; j++){
      for (i=is; i<ie; i++){
	realI[index] = (float) I[k][j][i];//*kappa[i][j]; //\bar{q}=qk density in computational space
	realJ[indexJ] += (float) I[k][j][i]*pw[k];
	//	printf("pw[%d] =%lf\n",k,pw[k]); 
	index++;
	indexJ++;
      }
    }
  }
  /* Copy the radiation energy field from first angular bin to all angular bins */
  indexJ=nx1_r*nx2_r; 
  for (k=ks+1; k<ke; k++){
    for (j=0; j<nx2_r; j++){
      for (i=0; i<nx1_r; i++){
	realJ[indexJ] = realJ[j*nx1_r+i]; 
	indexJ++;
      }
    }
  }
  sprintf(filename,"rte-000.vtk"); 
  write_curvilinear_mesh(filename,1,dims, pts, nvars,vardims, centering, varnames, vars);

#ifdef TEST3
  printf("Emitting cell:\n"); 
  printf("r[%d] = %lf phi[%d] = %lf\n",is + 1*nx1_r/3,x1[is + 1*nx1_r/3],js + 1*nx2_r/4,x2[js + 1*nx2_r/4]);
  printf("(x,y) = (%lf,%lf)\n",x[is + 1*nx1_r/3][js + 1*nx2_r/4],y[is + 1*nx1_r/3][js + 1*nx2_r/4]);
#endif  

  /*-----------------------*/
  /* Main timestepping loop */
  /*-----------------------*/
  for (n=1; n<nsteps; n++){
    /*Spatial boundary conditions */
    //bcs are specified along a computational coord direction, but are a function of the physical coordinate of adjacent "real cells"
    for(l=0;l<num_ghost; l++){
      for(k=ks;k<ke; k++){
	for (j=js; j<je; j++){
	  I[k][j][l] = bc_x1i(x[j][is],y[j][is],xa1[k],n*dt);
#if defined(DELTA_ANGLE_C)	  
	  I[k][j][nx1-1-l] = (double) bc_x1f_polar_delta(x2[j], xa1_b[k], xa1_b[k+1], dxa1, n*dt);
	  //	  printf("%lf %lf
	  //fmod(fabs(DELTA_ANGLE_C - xa1_i),2*M_PI) <= dxa1 &&  fmod(fabs(DELTA_ANGLE_C - xa1_f),2*M_PI) <= dxa1)
#else
	  I[k][j][nx1-1-l] = (double) bc_x1f_polar(x2[j], xa1[k], n*dt);
#endif
	  //	  I[k][j][nx1-1-l] = bc_x1f(x[j][ie-1],y[j][ie-1],xa1[k],n*dt);
	}
	for (i=is; i<ie; i++){
	  if(X2_PERIODIC){
	    I[k][l][i] = I[k][je-1-l][i];
	    I[k][nx2-1-l][i] = I[k][js+l][i];
	  }
	  else{ 
	    I[k][l][i] = bc_x2i(x[js][i],y[js][i],xa1[k],n*dt);
	    I[k][nx2-1-l][i] = bc_x2f(x[je-1][i],y[je-1][i],xa1[k],n*dt);
	  }  
	} 
      }
    }
    /* set any fixed points inside domain */
    double fixed_I; 
    for (k=ks; k<ke; k++){
      for (j=js; j<je; j++){
	for (i=is; i<ie; i++){
#ifdef TEST3
	  if (i == is + 1*nx1_r/3 && j == js + 1*nx2_r/4){ //integer division
	    I[k][j][i] = 1.0;
	    // printf("U,V,W = %lf,%lf,%lf\n",U[k][j][i],V[k][j][i],W[k][j][i]);
	    //printf("inexact U,V,W = %lf,%lf,%lf\n",mu[k][0],mu[k][1],-mu[k][1]/kappa[k][j][i]);
	  }
	
#endif 
	  fixed_I = bc_interior(x1[i], x2[j], xa1[k], n*dt);
	  if (fixed_I != 0.0)
	    I[k][j][i] = fixed_I; 
	}
      }
    }

    double flux_limiter =0.0; 
    double flux_limiter_x1_l,flux_limiter_x1_r,flux_limiter_x2_l,flux_limiter_x2_r,flux_limiter_xa1_l,flux_limiter_xa1_r; 
    double *imu = (double *) malloc(sizeof(double)*3); //manually copy array for computing slope limiters
    double *pr = (double *) malloc(sizeof(double)*3); //manually copy array for computing slope limiters
    /* Donor cell upwinding */
    for (k=ks; k<ke; k++){
      for (j=js; j<je; j++){
	for (i=is; i<ie; i++){
	  /* First coordinate */
	  U_plus = fmax(U[k][j][i],0.0); // max{U_{i-1/2,j,m},0.0} LHS boundary
	  U_minus = fmin(U[k][j][i+1],0.0); // min{U_{i+1/2,j,m},0.0} RHS boundary
	  /* First order fluxes: F_i+1/2 - F_i-1/2 */
	  //	  net_flux[k][j][i] = dt/(kappa[k][j][i]*dx1)*(x1_b[i+1]*(fmax(U[k][j][i+1],0.0)*I[k][j][i] + U_minus*I[k][j][i+1])-x1_b[i]*(U_plus*I[k][j][i-1] + fmin(U[k][j][i],0.0)*I[k][j][i]));
	  //rewrite for bruls discretization
#if !defined(SECOND_ORDER)
	  net_flux[k][j][i] = dt/(vol[k][j][i])*(x1_b[i+1]*dxa1[k]*dx2*(fmax(U[k][j][i+1],0.0)*I[k][j][i] + U_minus*I[k][j][i+1])-x1_b[i]*dxa1[k]*dx2*(U_plus*I[k][j][i-1] + fmin(U[k][j][i],0.0)*I[k][j][i]));
#else
	  /* Second order fluxes */
	  if (U[k][j][i+1] > 0.0){ //middle element is always the upwind element
	    imu[0] = I[k][j][i-1];
	    imu[1] = I[k][j][i];  
	    imu[2] = I[k][j][i+1];  
	    pr[0] = kappa[k][j][i-1];
	    pr[1] = kappa[k][j][i];  
	    pr[2] = kappa[k][j][i+1];  
	  }
	  else{
	    imu[0] = I[k][j][i+2];
	    imu[1] = I[k][j][i+1];  
	    imu[2] = I[k][j][i];  
	    pr[0] = kappa[k][j][i+2];
	    pr[1] = kappa[k][j][i+1];  
	    pr[2] = kappa[k][j][i];  
	  }
	  flux_limiter_x1_r= flux_PLM(dx1,imu);
	  flux_limiter_x1_r = flux_PLM_athena(pr, 1, dt, dx1, fabs(U[k][j][i+1]), imu);

	  //F^H_{i+1/2,j}
	  //net_flux[k][j][i] -= dt/(kappa[k][j][i]*dx1)*(x1_b[i+1]*(1-dt*fabs(U[k][j][i+1])/(dx1))*fabs(U[k][j][i+1])*flux_limiter_x1_r/2);

	  if (U[k][j][i] > 0.0){
	    imu[0] = I[k][j][i-2];  //points to the two preceeding bins; 
	    imu[1] = I[k][j][i-1];  
	    imu[2] = I[k][j][i];  
	    pr[0] = kappa[k][j][i-2];  //points to the two preceeding bins; 
	    pr[1] = kappa[k][j][i-1];  
	    pr[2] = kappa[k][j][i];  
	  }
	  else{
	    imu[0] = I[k][j][i+1]; //centered around current bin
	    imu[1] = I[k][j][i];  
	    imu[2] = I[k][j][i-1];  
	    pr[0] = kappa[k][j][i+1]; //centered around current bin
	    pr[1] = kappa[k][j][i];  
	    pr[2] = kappa[k][j][i-1];  
	  }
	  flux_limiter_x1_l= flux_PLM(dx1,imu);
	  flux_limiter_x1_l = flux_PLM_athena(pr, 1, dt, dx1, fabs(U[k][j][i]), imu);
	  //F^H_{i-1/2,j}
	  //net_flux[k][j][i] += dt/(kappa[k][j][i]*dx1)*(x1_b[i]*(1-dt*fabs(U[k][j][i])/(dx1))*fabs(U[k][j][i])*flux_limiter_x1_l/2);
	  //	  net_flux[k][j][i] = dt/(kappa[k][j][i]*dx1)*(x1_b[i+1]*flux_limiter_x1_r*fabs(U[k][j][i+1]) - x1_b[i]*flux_limiter_x1_l*fabs(U[k][j][i]));

	  //working formula for copied flux_PLM
	  //	  net_flux[k][j][i] = dt/(kappa[k][j][i]*dx1)*(x1_b[i+1]*flux_limiter_x1_r*U[k][j][i+1] - x1_b[i]*flux_limiter_x1_l*U[k][j][i]);

	  //rewrite for bruls discretization
	  net_flux[k][j][i] = dt/(vol[k][j][i])*(x1_b[i+1]*dxa1[k]*dx2*flux_limiter_x1_r*U[k][j][i+1] - x1_b[i]*dxa1[k]*dx2*flux_limiter_x1_l*U[k][j][i]);

#endif
	  /* Second coordinate */
	  V_plus = fmax(V[k][j][i],0.0); // max{V_{i,j-1/2},0.0} LHS boundary
	  V_minus = fmin(V[k][j+1][i],0.0); // min{V_{i,j+1/2},0.0} RHS boundary
	  /* Fluxes: G_i,j+1/2 - G_i,j-1/2 */
	  //	  net_flux[k][j][i] += dt/(kappa[k][j][i]*dx2)*((fmax(V[k][j+1][i],0.0)*I[k][j][i] + V_minus*I[k][j+1][i])-(V_plus*I[k][j-1][i] + fmin(V[k][j][i],0.0)*I[k][j][i]));
	  //rewrite for bruls discretization
#if !defined(SECOND_ORDER)
	  net_flux[k][j][i] += dt/(vol[k][j][i])*(dx1*dxa1[k]*(fmax(V[k][j+1][i],0.0)*I[k][j][i] + V_minus*I[k][j+1][i])-dx1*dxa1[k]*(V_plus*I[k][j-1][i] + fmin(V[k][j][i],0.0)*I[k][j][i]));
#else
	  /* Second order fluxes */
	  if (V[k][j+1][i] > 0.0){
	    imu[0] = I[k][j-1][i];  //points to the two preceeding bins; 
	    imu[1] = I[k][j][i];  
	    imu[2] = I[k][j+1][i];  
	  }
	  else{
	    imu[0] = I[k][j+2][i];
	    imu[1] = I[k][j+1][i];  
	    imu[2] = I[k][j][i];  
	  }
	  flux_limiter_x2_r= flux_PLM(dx2,imu);
	  flux_limiter_x2_r = flux_PLM_athena(pr, 2, dt, kappa[k][j][i]*dx2, fabs(V[k][j+1][i]), imu);//pr isnt dereferenced if dir!=1
	  //G^H_{i,j+1/2}
	  //	  net_flux[k][j][i] -= dt/(kappa[k][j][i]*dx2)*((1-dt*fabs(V[k][j+1][i])/(kappa[k][j][i]*dx2))*fabs(V[k][j+1][i])*flux_limiter_x2_r/2);
	  if (V[k][j][i] > 0.0){
	    imu[0] = I[k][j-2][i];  //points to the two preceeding bins; 
	    imu[1] = I[k][j-1][i];  
	    imu[2] = I[k][j][i];  
	  }
	  else{
	    imu[0] = I[k][j+1][i]; //centered around current bin
	    imu[1] = I[k][j][i];  
	    imu[2] = I[k][j-1][i];  
	  }
	  flux_limiter_x2_l = flux_PLM(dx2,imu);
	  flux_limiter_x2_l = flux_PLM_athena(pr, 2, dt, kappa[k][j][i]*dx2, fabs(V[k][j][i]), imu);
	  //G^H_{i,j-1/2}
	  //	  net_flux[k][j][i] += dt/(kappa[k][j][i]*dx2)*((1-dt*fabs(V[k][j][i])/(kappa[k][j][i]*dx2))*fabs(V[k][j][i])*flux_limiter_x2_l/2);

	  //working formula for copied flux_PLM
	  //	  net_flux[k][j][i] += dt/(kappa[k][j][i]*dx2)*(flux_limiter_x2_r*V[k][j+1][i] - flux_limiter_x2_l*V[k][j][i]);

	  //rewrite for bruls discretization
	  net_flux[k][j][i] += dt/(vol[k][j][i])*(dx1*dxa1[k]*flux_limiter_x2_r*V[k][j+1][i] - dxa1[k]*dx1*flux_limiter_x2_l*V[k][j][i]);
#endif

	  /* Third coordinate */
	  /* Have to manually compute indices due to lack of ghost cells */
	  /* Fluxes: H_i,j,n+1/2 - H_i,j,n-1/2 */
	  if (k==(ke-1)){
	    prev = k-1; 
	    next = ks;
	  } 
	  else if(k==ks){
	    prev = ke-1;
	    next=k+1; 
	  }
	  else{
	    prev = k-1;
	    next = k+1;
	  }
	  //need another position for van Leer slopes
	  if (prev==ks){ //these cases might not cover every situation
	    prev2 = ke-1;
	    next2 = next+1;
	  }
	  else if (next ==ke-1){
	    next2 = ks; 
	    prev2 = prev-1;
	  }
	  else{
	    next2 = next+1;
	    prev2 = prev-1;
	  }
	  W_plus = fmax(W[k][j][i],0.0); 
	  W_minus = fmin(W[next][j][i],0.0); 

#if !defined(SECOND_ORDER)
	  //	  net_flux[k][j][i] += (dt/(dxa1)*((fmax(W[next][j][i],0.0)*I[k][j][i] + W_minus*I[next][j][i])-(W_plus*I[prev][j][i] + fmin(W[k][j][i],0.0)*I[k][j][i])));
	  //rewrite for bruls discretization
	  net_flux[k][j][i] += (dt/(vol[k][j][i])*(kappa[next][j][i]*dx1*dx2*(fmax(W[next][j][i],0.0)*I[k][j][i] + W_minus*I[next][j][i])-kappa[k][j][i]*dx1*dx2*(W_plus*I[prev][j][i] + fmin(W[k][j][i],0.0)*I[k][j][i])));
#else
	  /* Second order fluxes */
	  if (W[next][j][i] > 0.0){
	    imu[0] = I[prev][j][i];  //points to the two preceeding bins; 
	    imu[1] = I[k][j][i];  
	    imu[2] = I[next][j][i];  
	  }
	  else{
	    imu[0] = I[next2][j][i]; //problem with no ghost cells
	    imu[1] = I[next][j][i];  
	    imu[2] = I[k][j][i];  
	  }
	  //	  flux_limiter_xa1_r= flux_PLM(dxa1,imu);
	  //correct dxa1[] element? 
	  flux_limiter_xa1_r = flux_PLM_athena(pr, 3, dt, fmod(fabs(xa1[k]-xa1[next]),2*M_PI), fabs(W[next][j][i]), imu);//pr isnt dereferenced if dir!=1
	  //H^H_{i,j+1/2}
	  //	  net_flux[k][j][i] += dt/(dxa1)*((1-dt*fabs(W[next][j][i])/(dxa1))*fabs(W[next][j][i])*flux_limiter_xa1_r/2);

	  if (W[k][j][i] > 0.0){
	    imu[0] = I[prev2][j][i];  //points to the two preceeding bins; 
	    imu[1] = I[prev][j][i];  
	    imu[2] = I[k][j][i];  
	  }
	  else{
	    imu[0] = I[next][j][i]; //centered around current bin
	    imu[1] = I[k][j][i];  
	    imu[2] = I[prev][j][i];  
	  }
	  //	  flux_limiter_xa1_l = flux_PLM(dxa1,imu);
	  flux_limiter_xa1_l = flux_PLM_athena(pr, 3, dt, fmod(fabs(xa1[k]-xa1[prev]),2*M_PI), fabs(W[k][j][i]), imu);//pr isnt dereferenced if dir!=1
	  //H^H_{i,j-1/2}
	  //net_flux[k][j][i] -= dt/(dxa1)*((1-dt*fabs(W[k][j][i])/(dxa1))*fabs(W[k][j][i])*flux_limiter_xa1_l/2);
	  //working formula for copied flux_PLM
	  //	  net_flux[k][j][i] += dt/(dxa1)*(flux_limiter_xa1_r*W[next][j][i] - flux_limiter_xa1_l*W[k][j][i]);
	  //rewrite for bruls discretization
	  net_flux[k][j][i] += dt/(vol[k][j][i])*(kappa[next][j][i]*dx1*dx2*flux_limiter_xa1_r*W[next][j][i] - kappa[k][j][i]*dx1*dx2*flux_limiter_xa1_l*W[k][j][i]);
	  /*	  printf("k = %d prev = %d next = %d\n",k,prev,next);
		  printf("xa1 = %lf xa1_prev = %lf xa1_next = %lf dxa1_l = %lf dxa1_r = %lf \n",xa1[k],xa1[prev],xa1[next],fmod(fabs(xa1[k]-xa1[prev]),2*M_PI),fmod(fabs(xa1[k]-xa1[next]),2*M_PI)); */
#endif

#ifdef NEW_FLUX  //regroup fluxes in dimension. assume appropriate limited van leer slope has been calculated
	  //why should signs be flipped on correction fluxes??
	  net_flux[k][j][i] = dt/(kappa[k][j][i]*dx1*dx2*dxa1)*(x1_b[i+1]*dx2*dxa1*(fmax(U[k][j][i+1],0.0)*I[k][j][i] + U_minus*I[k][j][i+1] - (1-dt*fabs(U[k][j][i+1])/(dx1))*fabs(U[k][j][i+1])*flux_limiter_x1_r/2)- //A_{i+1/2}F_{i+1/2}
								x1_b[i]*dx2*dxa1*(U_plus*I[k][j][i-1] + fmin(U[k][j][i],0.0)*I[k][j][i] - (1-dt*fabs(U[k][j][i])/(dx1))*fabs(U[k][j][i])*flux_limiter_x1_l/2) //A_{i-1/2}F_{i-1/2}
								+dx1*dxa1*(fmax(V[k][j+1][i],0.0)*I[k][j][i] + V_minus*I[k][j+1][i] - ((1-dt*fabs(V[k][j+1][i])/(kappa[k][j][i]*dx2))*fabs(V[k][j+1][i])*flux_limiter_x2_r/2)) //A_{j+1/2}G_{j+1/2}
								-dx1*dxa1*(V_plus*I[k][j-1][i] + fmin(V[k][j][i],0.0)*I[k][j][i] - (1-dt*fabs(V[k][j][i])/(kappa[k][j][i]*dx2))*fabs(V[k][j][i])*flux_limiter_x2_l/2)//A_{j-1/2}G_{j-1/2}
								+kappa[k][j][i]*dx1*dx2*(fmax(W[next][j][i],0.0)*I[k][j][i] + W_minus*I[next][j][i] - ((1-dt*fabs(W[next][j][i])/(dxa1))*fabs(W[next][j][i])*flux_limiter_xa1_r/2)) //A_{l+1/2}H_{l+1/2}
								-kappa[k][j][i]*dx1*dx2*(W_plus*I[prev][j][i] + fmin(W[k][j][i],0.0)*I[k][j][i] - ((1-dt*fabs(W[k][j][i])/(dxa1))*fabs(W[k][j][i])*flux_limiter_xa1_l/2))); //A_{l-1/2}H_{l-1/2}
#endif
	}
      }
    }

    /*Apply fluxes */
      for (k=ks; k<ke; k++){
	for (j=js; j<je; j++){
	  for (i=is; i<ie; i++){
	    I[k][j][i] -= net_flux[k][j][i];
	}
      }
    }
    
    /*Source terms */
      for (k=ks; k<ke; k++){
	for (j=js; j<je; j++){
	  for (i=is; i<ie; i++){
	    I[k][j][i] += source[k][j][i]*I[k][j][i]*dt;
	  }
	}
      }

    /*Output */
    //for now, explicitly copy subarray corresponding to real zonal info:
    index=0; 
    //zero out J array:
    indexJ=0;
    for (j=0; j<nx2_r; j++){
      for (i=0; i<nx1_r; i++){
	realJ[indexJ] = 0;
	indexJ++;
      }
    }

    for (k=ks; k<ke; k++){
      indexJ =0; 
      for (j=js; j<je; j++){
	for (i=is; i<ie; i++){
	  //index =(j-num_ghost)*nx2_r + (i-num_ghost); 
	  realI[index] = (float) I[k][j][i];//*kappa[i][j]; //\bar{q}=qk density in computational space
	  /*compute zeroth angular moment of the radiation field*/
	  realJ[indexJ] += (float) I[k][j][i]*pw[k];
	  indexJ++; 
	  index++;
	}
      }
    }
    /* Copy the radiation energy field from first angular bin to all angular bins */
    indexJ=nx1_r*nx2_r; 
    for (k=ks+1; k<ke; k++){
      for (j=0; j<nx2_r; j++){
	for (i=0; i<nx1_r; i++){
	  realJ[indexJ] = realJ[j*nx1_r+i]; 
	  indexJ++;
	}
      }
    }
   
    sprintf(filename,"rte-%.3d.vtk",n); 
    if(!OUTPUT_INTERVAL){
      if (n==nsteps-1) //for only the final result
	write_curvilinear_mesh(filename,3,dims, pts, nvars,vardims, centering, varnames, vars);}
    else{
      if (!(n%OUTPUT_INTERVAL)) 
	write_curvilinear_mesh(filename,3,dims, pts, nvars,vardims, centering, varnames, vars);}
      
      printf("step: %d time: %lf max{I} = %0.7lf min{I} = %0.7lf sum{I} = %0.7lf \n",
	     n,n*dt,find_max(realI,nxa1*nx1_r*nx2_r),find_min(realI,nxa1*nx1_r*nx2_r),sum(realI,nxa1*nx1_r*nx2_r));
  }

  return(0); 
}

/* Map to physical (cartesian) coordinates */
double X_physical(double x1, double x2){
  double x_cartesian = x1*cos(x2); 
  return(x_cartesian); 
}
double Y_physical(double x1, double x2){
  double y_cartesian = x1*sin(x2); 
  return(y_cartesian); 
}

double initial_condition(double x, double y, double xa1){
  /*  if (x >= 1.0 && x <= 2.0 && y >= 1.0 && y <=2.0 && xa1 >= 3*M_PI/2){
    return(1.0);
    }*/
  return(0.0);
}

//for 2D polar coordinates:
//bc at innermost radius
double bc_x1i(double x, double y, double xa1, double t){
  return(0.0);
}
//bc at outermost radius
double bc_x1f(double x, double y, double xa1, double t){
  if ((x>1.5) && (x<=2.0) && (y>1.5) && (y<=2.0) && xa1 >= XA1_I && xa1 <= XA1_F ){
    return(1.0);
  }
  return(0.0);
}
//this bc is specified in terms of polar coordinates
float bc_x1f_polar(double phi, double xa1, double t){
  //specify in terms of Cartesian angles
#ifdef TEST1
  if (phi >= PHI_I && phi <= PHI_F && xA_coordinate_to_physical(xa1, phi) >= XA1_I_C && xA_coordinate_to_physical(xa1, phi) <= XA1_F_C )
    return(1.0);
#endif
  return(0.0);
}
//if you want to fix a point inside the domain at a particular intensity 
double bc_interior(double r, double phi, double xa1, double t){

#ifdef TEST2
  if (r >=1.75 && r <=2.0 && phi >= 5*M_PI/4 && phi <=11*M_PI/8)
    return(1.0);
#endif 

  return(0.0);
}
#ifdef DELTA_ANGLE_C
float bc_x1f_polar_delta(double phi, double xa1_i, double xa1_f, double dxa1, double t){
  printf("phi = %lf, xa1_i = %lf, xa1_f = %lf xa1_i_c = %lf xa1_f_c = %lf dxa_l =%lf dxa_r = %lf\n",phi,xa1_i,xa1_f,xA_coordinate_to_physical(xa1_i,phi),xA_coordinate_to_physical(xa1_f,phi),fmod(fabs(DELTA_ANGLE_C - xA_coordinate_to_physical(xa1_i,phi)),2*M_PI), fmod(fabs(DELTA_ANGLE_C - xA_coordinate_to_physical(xa1_f,phi)),2*M_PI));
  if (phi >= PHI_I && phi <= PHI_F && fmod(fabs(DELTA_ANGLE_C - xA_coordinate_to_physical(xa1_i,phi)),2*M_PI) <= dxa1 &&  fmod(fabs(DELTA_ANGLE_C - xA_coordinate_to_physical(xa1_f,phi)),2*M_PI) <= dxa1){
    printf("phi = %lf, xa1_i = %lf, xa1_f = %lf\n",phi,xa1_i,xa1_f);
    return(1.0);
  }
  return(0.0);
}
#endif
//bc at phi=0.0
double bc_x2i(double x, double y, double xa1, double t){
  return(0.0);
}
//bc at phi_final
double bc_x2f(double x, double y, double xa1, double t){
  return(0.0);
}

float find_max(float a[], int n) {
  int i,index;
  float max; 
  max = a[0];
  index = 0;
  for (i = 1; i < n; i++) {
    if (a[i] > max) {
      index = i;
      max = a[i];
    }
  }
  return(max); 
}

double find_max_double(double a[], int n) {
  int i,index;
  double max; 
  max = a[0];
  index = 0;
  for (i = 1; i < n; i++) {
    if (a[i] > max) {
      index = i;
      max = a[i];
    }
  }
  return(max); 
}

float find_min(float a[], int n) {
  int i,index;
  float min; 
  min = a[0];
  index = 0;
  for (i = 1; i < n; i++) {
    if (a[i] < min) {
      index = i;
      min = a[i];
    }
  }
  return(min); 
}

float sum(float a[], int n) {
  int i,index;
  float sum; 
  sum = a[0];
  for (i = 1; i < n; i++) {
    sum+= a[i]; 
  }
  return(sum); 
}

/*Transform polar vector (i.e. edge velocity) from orthonormal local basis to physical basis */
void vector_coordinate_to_physical(double vr, double vphi, double vphase, double phi, double r, double *vx, double *vy, double *vz){
  *vx = vr*cos(phi) -sin(phi)*vphi*r; 
  *vy = vr*sin(phi) +cos(phi)*vphi*r; 
  *vz = vphase; 
  return;
}

/*Transform Cartesian vector to orthonormal local basis */
void vector_physical_to_coordinate(double vx, double vy, double phi, double r, double *vr, double *vphi){
  *vr = vx*cos(phi) +sin(phi)*vy; 
  *vphi = (-vx*sin(phi) +cos(phi)*vy)/r; 
  return;
}

void velocity_physical(double x_b,double y_b,double *vx,double *vy){
  double angle = atan2(y_b,x_b); 
  *vx = -cos(angle);
  *vy = -sin(angle);
  return; 
}

void velocity_coordinate(double r_b,double phi_b,double *vr,double *vphi){

  return;
}

/* a duplication of the function in ATHENA FullRT_flux.c */
double flux_PLM(double ds,double *imu){
  //the upwind slope
  double delq1 = (imu[2] - imu[1])/ds;
  double delq2 = (imu[1] - imu[0])/ds;
  double dqi0;
  if(delq1*delq2 >0.0) //minmod function
    dqi0 = 2.0*delq1*delq2/(delq1+delq2);
  else
    dqi0 = 0.0;
  //unknown why ds is in this function. might have to do with nonuniform grids
  return(ds*dqi0); 
}

int uniform_angles2D(int N, double phi_z, double *pw, double **mu, double **mu_b, double *xa1,double *xa1_b){
/* Generate uniform 2D discretization of mu */

  //   Input: N-- Number of mu level cells in one \hat{k}^i dimension
  //  (note, there are n+1, n/2+1 boundaries, including -1,1 and 0,1) N MUST BE EVEN, should be greater than 6
  //   phi: angle down from the z-axis that fixes the plane of propagation directions. pi/2 for k_z=0

  //   Output: 
  //   nxa: angular parameter coordinate dimensions. currently, it is 2-dim with nxa(1) = N nxa(2) = N +1.   

  // The algorithm discretizes theta \in [0, 2pi ) then generates the boundary rays mu_b from these \theta
  // The actual sampling rays are generated by averaging the \theta of the boundary rays and then 
  // using spherical polar coordinates to convert to k^x, k^y, k^z

  // Keep data structures same as 3D case so that 2D problems are a subset									  
  int i,j,k;
  int nxa1 = N;
  double  dxa1 = 2*M_PI/(nxa1); //dont mesh all the way to 2pi
  xa1_b[0] = 0.0;
  xa1[0] = dxa1/2;
  for (i=1; i<nxa1; i++){
    xa1_b[i] = xa1_b[i-1] + dxa1; 
    xa1[i] = xa1[i-1] + dxa1; 
  }

  //add another boundary ray for purposes of VTK output
  xa1_b[nxa1]  = xa1_b[nxa1-1] + dxa1; 
 
  for (i=0; i<= nxa1; i++){
    //  for (i=0; i< nxa1; i++){
    mu_b[i][0] = cos(xa1_b[i])*sin(phi_z);
    mu_b[i][1] = sin(xa1_b[i])*sin(phi_z);
    mu_b[i][2] = cos(phi_z);
  }
  //periodicity of the domain implies duplication of 1 theta bin first and last column/row are identical for kx, ky
  for (i=0; i<nxa1; i++){
    mu[i][0] = cos(xa1[i])*sin(phi_z);
    mu[i][1] = sin(xa1[i])*sin(phi_z);
    mu[i][2] = cos(phi_z);
  }

  /*------------------CALCULATE QUADRATURE WEIGHTS ------------------- */
  // Should this be proportional to the "size of the cell"?
  for (i=0; i<nxa1; i++)
    pw[i] = 1.0/(nxa1);
  return(nxa1);  
}

/* Function for returning pointer that allows A[i][j] access of array 
   such that i increases first in memory, then second index */
// pass in contiguous chunk of memory of size n1*n2*n3
// creates many excess pointers ....
double **allocate_2D_contiguous(double *a, int n1, int n2){ //this is reordered since advection
  int i; 
  double **a_p = (double **) malloc(sizeof(double *)*n1);
  for(i=0; i<n1; i++)
    a_p[i] = malloc(sizeof(double)*n2); //&(a[n2*i]);    
  return(a_p); 
}

/* Function for returning pointer that allows A[][][] access of array */
double ***allocate_3D_contiguous(double *a, int n1, int n2, int n3){
  int i,j; 
  double ***a_p = (double ***) malloc(sizeof(double **)*n1);
  double **a_p2 = (double **) malloc(sizeof(double *)*n2*n1);
  for(i=0; i<n1; i++){
    a_p[i] = (double **) malloc(sizeof(double *)*n2); //&(a_p2[n2*n3*i]);
    for (j=0; j< n2; j++){
      a_p[i][j] = malloc(sizeof(double)*n3); //&(a[n2*n3*i + n3*j]);    
    }
  } 
  return(a_p);
}

float analytic_solution(double x,double y, double angle_c, double r_max){ //true radius, not cell centered r
  //is there a way to do this without ray tracing?

  //worry about the donut hole?

  //nonlinear root finding

			    
  return(0.0);
}

double newton_raphson(double x, double y, double r_max, double angle_c, double x0, double allerr, int maxmitr){ //later add general function pointers
  int itr; 
  double h, x1;
  for (itr=1; itr<=maxmitr; itr++){
    h=f(x0,x,y,angle_c,r_max)/df(x0,x,y,r_max);
    x1=x0-h;
    if (fabs(h) < allerr)
      return x1;
    x0=x1;
    //    printf("phi = %lf\n",x0);
  }
  printf(" The required solution does not converge or iterations are insufficient\n");
  return 1;
} 

double f(double phi, double x, double y,double r_max, double angle_c){
  return tan(angle_c) - (y-r_max*sin(phi))/(x-r_max*cos(phi)); 
}

double df(double phi, double x, double y, double r_max){
  return r_max*cos(phi)/(x-r_max*cos(phi)) - (r_max*sin(phi)*(r_max*sin(phi)-y))/pow((x-r_max*cos(phi)),2);
}

double xA_coordinate_to_physical(double xa1, double x2){
  return fmod(xa1 + x2,2*M_PI);
} 

double flux_PLM_athena(double r[3], int dir, double dt, double ds, double vel, double imu[3])
{
  /* for each ray, we always use the upwind specific intensity , therefore velocity is always positive */
  /* imup[0:2] is i-2, i-1, i*/

  double dqi0, delq1, delq2;
  double *imup;
  double distance;
  double *pr;
  pr = &(r[2]);
  double geom1 = 1.0, geom2 = 1.0;
  double geom3 = 1.0, geom4 = 1.0; /* geometric weighting factor */

  imup = &(imu[2]);

  /* The upwind slope */
  delq1 = (imup[0] - imup[-1]) / ds;
  delq2 = (imup[-1] - imup[-2]) / ds;
  /* Only need to apply the weighting factor when we want to calculate
   * flux along radial direction */
  if (dir ==1 ){
    geom1 = 1.0/(1.0 - ds * ds/(12.0 * pr[0] * pr[-1]));
    geom2 = 1.0/(1.0 - ds * ds/(12.0 * pr[-1] * pr[-2]));
    
    delq1 *= geom1;
    delq2 *= geom2;
  }

  if(delq1 * delq2 > 0.0){
    dqi0 = 2.0 * delq1 * delq2 / (delq1 + delq2);
  }
  else{
    dqi0 = 0.0;
  }

  /* Take into account the curvature effect for time averaged interface state */
  /* See eq. 64 of skinner & ostriker 2010 */
  if (dir == 1){
    geom3 = 1.0 - ds /(6.0 * pr[-1]);
    geom4 = 1.0 - vel * dt/(6.0*(0.5 * (pr[-1] + pr[0]) - 0.5 * vel * dt));
  }

  distance = ds * geom3;
  distance -= ((vel * dt) * geom4);

  /* The upwind flux */
  return(imup[-1] + distance * (dqi0/2.0)); 
}
int bruls_angles2D(int N, double phi_z, double *pw, double **mu, double **mu_b, double *xa1,double *xa1_b){
  int num_rays = N*(N+2)/2; //up to 84 in 2D
  int num_rays_per_quadrant = num_rays/4; 


  //made modifications relative to MATLAB code to force phi_z=pi
  double *ordinates = malloc(sizeof(double)*N/2); 
  ordinates[0] = sqrt(1.0/(3.0*(N-1))); //changed
  double dcos = (1-3*pow(ordinates[0],2))*2/(N-2); //changed 

  int i, j,k,l,m;
  /* Choice of first cosine is arbitrary. In analogy to Gaussian quadrature */
  for (i=0; i<N/2; i++){
    ordinates[i] = sqrt(ordinates[0]*ordinates[0] + i*dcos); 
  }

  // Derive weights for integration. Ref: Bruls 1999 appendix
  double *W = malloc(sizeof(double)*N/2); 
  for (i=0; i<N/2; i++){
    W[i] = sqrt(4.0/(3.0*(N-1)) + i*2.0/(N-1));  //should I change?
    //    printf("W[%d] = %lf\n",i,W[i]); 
  }
  //compute level weights
  double *lw = malloc(sizeof(double)*N/2); 
  lw[0] = W[0];
  double sum =lw[0]; 
  for (i=1; i<N/2-1; i++){
    lw[i] = W[i] - W[i-1];
    sum += lw[i]; 
  }

  /* following ATHENA, we correct the last level weight to renormalize...not
     sure why this happens */
  lw[N/2-1] = 1.0 - sum; 
  
  /* Permutation matrix */
  double **pmat; 
  double *data_pmat = malloc(sizeof(double)*N/2*N/2); 

  pmat = allocate_2D_contiguous(data_pmat,N/2,N/2); 
  int *plab; 
  plab = malloc(sizeof(int *)*num_rays); 
  
  int **pl = malloc(sizeof(int *)*N/2); 
  //initialize pl to zeros
  for (i=0; i<N/2; i++)
    pl[i] = malloc(sizeof(int)*3); 
  /*  for (i=0; i<N/2; i++)
    for (j=0; j<3; j++)
      pl[i][j] = 0;  */

  int ray_count = 0; 
  int np =0; 
  int ip =0; 
  /* To select valid rays for the quadrature, we follow the index selection
  rule: sigma = i + j + k
  i.e. indices of the direction cosines must equal ntheata/2 +2
  for proper normalization in 3D space */
  for (i=0; i<N/2; i++){
    for (j=0; j<N/2-i; j++){
      k = N/2-1-i-j; //assume this is correct 
      mu[ray_count][0] = ordinates[i]; 
      mu[ray_count][1] = ordinates[j]; 
      ip = permutation(i,j,k,pl,np); 
      /*      printf("ip = %d np = %d i,j,k = %d,%d,%d\n",ip,np, i,j,k); 
      printf("pl =\n");
      for (l=0; l<N/2; l++){
	for (m=0; m<3; m++)
	  printf("%d ", pl[l][m]); 
	printf("\n");
	} */
      if (ip == -1){ //ray indices havent been loaded yet
	pl[np][0] = i; 
	pl[np][1] = j; 
	pl[np][2] = k; 
	pmat[i][np]++; 
	plab[ray_count] = np; 
	np = np+1; 
      }
      else {
	pmat[i][ip]++;
	plab[ray_count] = ip; 	     
      }
      ray_count++;  
    }
  }
  assert(ray_count == num_rays_per_quadrant); 

  /* discretization symmetry: reflect across second axis */
  for (i=0; i<ray_count; i++){
    mu[ray_count+i][0] = -mu[i][0]; 
    mu[ray_count+i][1] = mu[i][1]; 
  }
  ray_count*=2;  
  /* discretization symmetry: reflect across second axis */
  //flip the order for this so that the rays are in order of xa1
  for (i=0; i<ray_count; i++){
    mu[2*ray_count-(i+1)][0] = mu[i][0]; 
    mu[2*ray_count-(i+1)][1] = -mu[i][1]; 
  }
  ray_count*=2; 

  /* Solve system of equations to calculate families of point weights */
  double *wpf = malloc(sizeof(double)*(N/2-1)); 
  /*  printf("pmat=\n");
  for (l=0; l<N/2; l++){
    for (m=0; m<N/2; m++)
      printf("%lf ", pmat[l][m]); 
    printf("\n");
  }
  printf("lw=\n");
  for (l=0; l<N/2; l++)
  printf("%lf ", lw[l]);  */

  gaussianelim(pmat, lw, wpf, N/2-1, 1);
  /*  for (i=0; i<N/2-1; i++)
    printf("wpf[%d] = %lf\n",i,wpf[i]);
  for (i=0; i<ray_count; i++)
    printf("plab[%d] = %d\n",i,plab[i]);
  */
  for(i=0; i<num_rays_per_quadrant; i++){
    pw[i] = wpf[plab[i]]/4; 
    pw[i+num_rays_per_quadrant] = wpf[plab[i]]/4; 
    pw[i+2*num_rays_per_quadrant] = wpf[plab[i]]/4; 
    pw[i+3*num_rays_per_quadrant] = wpf[plab[i]]/4; 
  }
  
  /* Compute angles corresponding to sample rays */
  for (i=0; i<num_rays; i++){
    xa1[i] = atan2(mu[i][1],mu[i][0]) + M_PI;
  }
  /* Sort by increasing xa1 */
  qsort(xa1, num_rays, sizeof(double), cmpfunc); 
  //better way of resorting mu?
  //recompute mu
  for (i=0; i < num_rays; i++){
    mu[i][0] = cos(xa1[i])*sin(phi_z); 
    mu[i][1] = sin(xa1[i])*sin(phi_z);
    mu[i][2] = cos(phi_z); 
  }

  /* Boundary angles lie halfway between adjacent sample rays */
  //i am assuming they are ordered in xi here, and:
  // the first ray is in the first quadrant
  // the last ray is in the last quadrant 
  xa1_b[0] = fmod((xa1[0] +2*M_PI + xa1[num_rays-1])/2, 2*M_PI); 
  for (i=1; i<num_rays; i++){
    xa1_b[i] = (xa1[i] + xa1[i-1])/2; 
  }
  //  xa1_b[num_rays] = xa1_b[0];  
  xa1_b[num_rays] = 2*M_PI; //can i assume this?
  for (i=0; i <= num_rays; i++){
    mu_b[i][0] = cos(xa1_b[i])*sin(phi_z); 
    mu_b[i][1] = sin(xa1_b[i])*sin(phi_z);
    mu_b[i][2] = cos(phi_z); 
  }
  return(num_rays); 
}
//stolen from ATHENA to calculate point weights
int permutation(int i, int j, int k, int ** pl, int np){
  int ip = -1;
  int l,m,n,o; 
  for (l=0; l<np; l++){
    for(m=0; m<3; m++){
      if (i == pl[l][m])
	for (n=0; n<3; n++){
	  if (n != m){
	    if (j == pl[l][n]){
	      for (o=0; o<3; o++){
		if(o!= m && o!= n){
		  if (k == pl[l][o]){
		    ip = l; 
		  }
		}
	      }
	    }
	  }
	}
    }
  }           
  return(ip); 
}

void gaussianelim(double **A, double *b, double *x, int n, int pivot) { //pivot 1 for partial row, 0 for no pivoting
  int i, j, pivot_pos, column;
  double **Ab; 
  double ratio,tmp;

  /* Allocate the new augmented matrix */
  Ab = (double **) malloc(sizeof(double)*(n));//allocate n pointers to rows of pointers
  for (i=0; i<n; i++){
    Ab[i] = (double *) malloc(sizeof(double)*(n+1));
  }

  /*Copy the old matrix and rhs into augmented matrix */
  for (i=0; i <n; ++i){     //rows
    for (j=0; j <n; ++j){ //collumns
      Ab[i][j] = A[i][j];
    }
  }
  for (i=0; i<n; i++) Ab[i][n] = b[i];

  /*Gaussian Elimination */
  /*Pivoting */
  double *temp;
  temp = (double *) calloc(sizeof(double),n+1);
  if (pivot) {
    for (j=0; j<n; j++){
      for (i=j+1; i<n; i++){ //check all rows beneath current pivot pos
	if (fabs(Ab[i][j]) > fabs(Ab[j][j])) {
	  temp = Ab[i];
	  Ab[i] = Ab[j];
	  Ab[j] = temp;
	}
      }
    }
  }
  
  /* Elimination of variables via basic row operations */
  for (i =0; i < (n-1); i++) {           //starting with first row, first column
    for (j=i+1; j<n; ++j) {              // take next row, first column
      ratio = Ab[j][i] / Ab[i][i];       //compute their ratio
      for (column = i; column <n+1; column++){    // and eliminate it from the second row
	Ab[j][column] -= (ratio * Ab[i][column]); // by subtracting it from all coef
      } //do this for all rows until the last row
    }
  }
  
  /* Back Subsitution */
  for (i=n-1; i>=0; i--){ //rows, starting from bottom
    tmp = 0; //Build a temporary dbl precision that holds the solved variables* the rows coefficients to subtract from RHS for each line
    for (j= i+1; j<n; j++) { //this inner loop doesnt do anything for the bottom row, since the equation req no more info
      tmp += x[j]*Ab[i][j];
    }
    x[i] = (Ab[i][n] - tmp)/ Ab[i][i]; //RHS - tmp (known variables, scaled) / unknown coeff
  }
}

int cmpfunc (const void * a, const void * b)
{
  double a_r = *(double*)a;
  double b_r = *(double*)b;
  if (a_r > b_r ){
    return(1);
  }
  else if (b_r > a_r){
    return(-1);
  }
  else {
    return(0);
  }
}
