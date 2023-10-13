from typing import Self
from ufl import (TestFunctions, TrialFunction, Identity, as_tensor, as_vector, eq, grad, det, div, dev, inv, tr, sqrt, conditional ,
                gt, inner, derivative, dot, ln, split,acos,cos,sin,lt,
                as_tensor, as_vector, SpatialCoordinate)
import ufl
from dolfinx.fem import Constant
from petsc4py import PETSc
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, Expression, locate_dofs_topological)

import numpy as np
#Class that decribes the thermoelastic axial strain problem
class GEL_Axial_Symmetric():


    def __init__(self,domain,**kwargs):

        #Create Contants
        self.Gshear_0= Constant(domain,1000.0)         # Shear modulus, kPa
        self.lambdaL = Constant(domain,100.0)            # Locking stretch
        self.Kbulk   = Constant(domain,1000.0*self.Gshear_0.__float__())  # Bulk modulus, kPa
        self.Omega   = Constant(domain,1.00e5)         # Molar volume of fluid
        self.D       = Constant(domain,5.00e-3)        # Diffusivity
        self.chi     = Constant(domain,0.1)            # Flory-Huggins mixing parameter
        self.theta0  = Constant(domain,298.0)            # Reference temperature
        self.R_gas   = Constant(domain,8.3145e6)       # Gas constant
        self.RT      = 8.3145e6*self.theta0 
        #
        self.phi0    = Constant(domain,0.999)          # Initial polymer volume fraction
        self.mu0     = ln(1.0-self.phi0) + self.phi0 + self.chi*self.phi0*self.phi0  #Initialize chemical potential 

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]


        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2) # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # For  pressure
        self.TH = ufl.MixedElement([self.U2, self.P1,self.P1,self.P1])     # Taylor-Hood style mixed element
        self.ME = FunctionSpace(domain, self.TH)    # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.p, self.mu , self.c = split(self.w)
        self.w_old         = Function(self.ME)
        self.u_old,  self.p_old, self.mu_old,self.c_old = split(self.w_old)

        self.u_test,self.p_test, self.mu_test,self.c_test= TestFunctions(self.ME)       
        self.dw = TrialFunction(self.ME)  

        
        self.x = SpatialCoordinate(domain)
        self.domain = domain

        
        #Initialize problem
        self.w.sub(2).interpolate(lambda x: np.full((x.shape[1],),  float(self.mu0)))
        self.w_old.sub(2).interpolate(lambda x: np.full((x.shape[1],),float(self.mu0)))

        c0 = 1/self.phi0 - 1
        self.w.sub(3).interpolate(lambda x: np.full((x.shape[1],),  float(c0)))
        self.w_old.sub(3).interpolate(lambda x: np.full((x.shape[1],),float(c0)))

    def WeakForms(self,dt):
        dk = Constant(self.domain,float(dt))
        dx = ufl.dx(metadata={'quadrature_degree': 4})

        # The weak form for the equilibrium equation
        Res_0 = inner(self.Tmat, self.ax_grad_vector(self.u_test) )*self.x[0]*dx

        # The weak form for the auxiliary pressure variable definition
        Res_1 = dot((self.p*self.Je/self.Kbulk + ln(self.Je)) , self.p_test)*self.x[0]*dx

        # The weak form for the mass balance of solvent      
        Res_2 = dot((self.c - self.c_old)/dk, self.mu_test)*self.x[0]*dx \
                -  self.Omega*dot(self.Jmat , self.ax_grad_scalar(self.mu_test) )*self.x[0]*dx
        
        # The weak form for the concentration
        fac = 1/(1+self.c)
        fac1 =  self.mu - ( ln(1.0-fac)+ fac + self.chi*fac*fac)
        fac2 = - (self.Omega*self.Je/self.RT)*self.p 
        fac3 = fac1 + fac2 
        #
        Res_3 = dot(fac3, self.c_test)*self.x[0]*dx
                
        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2 + Res_3
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        
        # Kinematics
        self.F = self.F_ax_calc(self.u)
        self.J = det(self.F)  # Total volumetric jacobian

        # Elastic volumetric Jacobian
        self.Je     = self.Je_calc(self.u,self.c)                    
        self.Je_old = self.Je_calc(self.u_old,self.c_old)

        #  Normalized Piola stress
        self.Tmat = self.Piola_calc(self.u, self.p)

        #  Normalized species  flux
        self.Jmat = self.Flux_calc(self.u, self.mu, self.c)



    def ax_grad_vector(self,u):
        grad_u = grad(u)
        return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                    [grad_u[1,0], grad_u[1,1], 0],
                    [0, 0, u[0]/self.x[0]]]) 

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def ax_grad_scalar(self,y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.])

    # Axisymmetric deformation gradient 
    def F_ax_calc(self,u):
        dim = len(u)
        Id = Identity(dim)          # Identity tensor
        F = Id + grad(u)            # 2D Deformation gradient
        F33 = 1+(u[0])/self.x[0]      # axisymmetric F33, R/R0    
        F33 = conditional(eq(self.x[0],0),1,F33)
        return as_tensor([[F[0,0], F[0,1],0.0],
                    [F[1,0], F[1,1], 0.0],
                    [0.0,0.0, F33]]) # Full axisymmetric F

    def lambdaBar_calc(self,u):
        F = self.F_ax_calc(u)
        C = F.T*F
        I1 = tr(C)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta
    
    def zeta0_calc(self):
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
        z    = 1/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Keep from blowing up
        beta0 = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta0 = (self.lambdaL/3)*beta0
        return zeta0

    #  Elastic Je
    def Je_calc(self,u,c):
        F = self.F_ax_calc(u)  
        detF = det(F)   
        #
        detFs = 1.0 + c          # = Js
        Je    = (detF/detFs)     # = Je
        return   Je    

    # Normalized Piola stress for Arruda_Boyce material
    def Piola_calc(self,u,p):
        F     = self.F_ax_calc(u)
        zeta  = self.zeta_calc(u)
        zeta0 = self.zeta0_calc()
        Tmat = (zeta*F - zeta0*inv(F.T) ) - self.J*p*inv(F.T)/self.Gshear_0
        return Tmat

    # Normalized species flux
    def Flux_calc(self,u, mu, c):
        F = self.F_ax_calc(u) 
        #
        Cinv = inv(F.T*F) 
        #
        Mob = (self.D*c)/(self.Omega*self.RT)*Cinv
        #
        Jmat = - self.RT* Mob * self.ax_grad_scalar(mu)
        return Jmat

class GEL_Plane_Strain():
    def __init__(self,domain,**kwargs):
       
        if "chi" in kwargs.keys():
            self.chi = kwargs["chi"]
        else:
            self.chi = Constant(domain,0.1)    

        if "Gshear" in kwargs.keys():
            self.Gshear_0 = kwargs["Gshear"]
        else:
            self.Gshear_0= Constant(domain,1000.0)

        if "Kbulk" in kwargs.keys():
            self.Kbulk = kwargs["Kbulk"]
        else:
            self.Kbulk   = self.Gshear_0*1000  # Bulk modulus, kPa
        
        if "D" in kwargs.keys():
            self.D = kwargs["D"]
        else:
             self.D       = Constant(domain,5.00e-3)  # Diffusivity

        if "Omega" in kwargs.keys():
            self.Omega = kwargs["Omega"]
        else:
            self.Omega   = Constant(domain,1.00e5)         # Molar volume of fluid
            
        

        #Create Contants
                # Shear modulus, kPa
        self.lambdaL = Constant(domain,1000.0)            # Locking stretch
       
               # Flory-Huggins mixing parameter
        self.theta0  = Constant(domain,298.0)            # Reference temperature
        self.R_gas   = Constant(domain,8.3145e6)       # Gas constant
        self.RT      = 8.3145e6*self.theta0 
        #
        self.phi0    = Constant(domain,0.999)          # Initial polymer volume fraction
        self.mu0     = ln(1.0-self.phi0) + self.phi0 #Initialize chemical potential 

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]



        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2) # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # For  pressure
        self.TH = ufl.MixedElement([self.U2, self.P1,self.P1,self.P1])     # Taylor-Hood style mixed element
        self.ME = FunctionSpace(domain, self.TH)    # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.p, self.mu , self.c = split(self.w)
        self.w_old         = Function(self.ME)
        self.u_old,  self.p_old, self.mu_old,self.c_old = split(self.w_old)

        self.u_test,self.p_test, self.mu_test,self.c_test= TestFunctions(self.ME)       
        self.dw = TrialFunction(self.ME)  

        
        self.x = SpatialCoordinate(domain)
        self.domain = domain


        #Initialize problem
        self.w.sub(2).interpolate(lambda x: np.full((x.shape[1],),  float(self.mu0)))
        self.w_old.sub(2).interpolate(lambda x: np.full((x.shape[1],),float(self.mu0)))

        c0 = 1/self.phi0 - 1
        self.w.sub(3).interpolate(lambda x: np.full((x.shape[1],),  float(c0)))
        self.w_old.sub(3).interpolate(lambda x: np.full((x.shape[1],),float(c0)))

        
    
    
    def WeakForms(self,dt):
        dk = Constant(self.domain,float(dt))
        dx = ufl.dx(metadata={'quadrature_degree': 4})

        # The weak form for the equilibrium equation
        Res_0 = inner(self.Tmat, self.pe_grad_vector(self.u_test) )*dx

        # The weak form for the auxiliary pressure variable definition
        Res_1 = dot((self.p*self.Je/self.Kbulk + ln(self.Je)) , self.p_test)*dx

        # The weak form for the mass balance of solvent      
        Res_2 = dot((self.c - self.c_old)/dk, self.mu_test)*dx \
                -  self.Omega*dot(self.Jmat , self.pe_grad_scalar(self.mu_test) )*dx
        
        # The weak form for the concentration
        fac = 1/(1+self.c)
        fac1 =  self.mu - ( ln(1.0-fac)+ fac + self.chi*fac*fac)
        fac2 = - (self.Omega*self.Je/self.RT)*self.p 
        fac3 = fac1 + fac2 
        #
        Res_3 = dot(fac3, self.c_test)*dx
                
        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2 + Res_3
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        
        # Kinematics
        self.F  = self.F_pe_calc(self.u)
        self.J = det(self.F)  # Total volumetric jacobian

        # Elastic volumetric Jacobian
        self.Je     = self.Je_calc(self.u,self.c)                    
        self.Je_old = self.Je_calc(self.u_old,self.c_old)

        #  Normalized Piola stress
        self.Tmat = self.Piola_calc(self.u, self.p)

        #  Normalized species  flux
        self.Jmat = self.Flux_calc(self.u, self.mu, self.c)
    
    def pe_grad_vector(self,u):
        grad_u = grad(u)
        return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def pe_grad_scalar(self,y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.])

    # Plane strain deformation gradient 
    def F_pe_calc(self,u):
        dim = len(u)
        Id = Identity(dim)          # Identity tensor
        F  = Id + grad(u)            # 2D Deformation gradient
        return as_tensor([[F[0,0], F[0,1], 0],
                    [F[1,0], F[1,1], 0],
                    [0, 0, 1]]) # Full pe F

    def lambdaBar_calc(self,u):
        F = self.F_pe_calc(u)
        C    = F.T*F
        I1   = tr(C)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta
    
    def zeta0_calc(self):
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        # This is sixth-order accurate.
        z    = 1/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Keep from blowing up
        beta0 = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta0 = (self.lambdaL/3)*beta0
        return zeta0

    def Je_calc(self,u,c):
        F = self.F_pe_calc(u)  
        detF = det(F)   
        #
        detFs = 1.0 + c
        Je    = (detF/detFs)
        return   Je    
    # Piola stress 
    def Piola_calc(self,u, p):
        F = self.F_pe_calc(u)
        J = det(F)
        #
        zeta = self.zeta_calc(u)
        zeta0 = self.zeta0_calc()
        #
       
        Tmat = (zeta*F - zeta0*inv(F.T) ) - J*p*inv(F.T)/self.Gshear_0
        return Tmat

    def Flux_calc(self,u, mu, c):
        F = self.F_pe_calc(u) 
        #
        Cinv = inv(F.T*F) 
        #
        Mob = (self.D*c)/(self.Omega*self.RT)*Cinv
        #
        Jmat = -self. RT* Mob * self.pe_grad_scalar(mu)
        return Jmat

   
class GEL_3D():
    def __init__(self,domain,**kwargs):

        #Create Contants
        self.Gshear_0= Constant(domain,1000.0)         # Shear modulus, kPa
        self.lambdaL = Constant(domain,100.0)            # Locking stretch
        self.Kbulk   = Constant(domain,1000.0*self.Gshear_0.__float__())  # Bulk modulus, kPa
        self.Omega   = Constant(domain,1.00e5)         # Molar volume of fluid
        self.D       = Constant(domain,5.00e-3)        # Diffusivity
        self.chi     = Constant(domain,0.1)            # Flory-Huggins mixing parameter
        self.theta0  = Constant(domain,298.0)            # Reference temperature
        self.R_gas   = Constant(domain,8.3145e6)       # Gas constant
        self.RT      = 8.3145e6*self.theta0 
        #
        self.phi0    = Constant(domain,0.999)          # Initial polymer volume fraction
        self.mu0     = ln(1.0-self.phi0) + self.phi0 + self.chi*self.phi0*self.phi0  #Initialize chemical potential 

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]


        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2) # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # For  pressure
        self.TH = ufl.MixedElement([self.U2, self.P1,self.P1,self.P1])     # Taylor-Hood style mixed element
        self.ME = FunctionSpace(domain, self.TH)    # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.p, self.mu , self.c = split(self.w)
        self.w_old         = Function(self.ME)
        self.u_old,  self.p_old, self.mu_old,self.c_old = split(self.w_old)

        self.u_test,self.p_test, self.mu_test,self.c_test= TestFunctions(self.ME)       
        self.dw = TrialFunction(self.ME)  

        
        self.x = SpatialCoordinate(domain)
        self.domain = domain

        
        #Initialize problem
        self.w.sub(2).interpolate(lambda x: np.full((x.shape[1],),  float(self.mu0)))
        self.w_old.sub(2).interpolate(lambda x: np.full((x.shape[1],),float(self.mu0)))

        c0 = 1/self.phi0 - 1
        self.w.sub(3).interpolate(lambda x: np.full((x.shape[1],),  float(c0)))
        self.w_old.sub(3).interpolate(lambda x: np.full((x.shape[1],),float(c0)))

    def WeakForms(self,dt):
        dk = Constant(self.domain,float(dt))
        dx = ufl.dx(metadata={'quadrature_degree': 4})

        # The weak form for the equilibrium equation
        Res_0 = inner(self.Tmat, grad(self.u_test) )*dx

        # The weak form for the auxiliary pressure variable definition
        Res_1 = dot((self.p*self.Je/self.Kbulk + ln(self.Je)) , self.p_test)*dx

        # The weak form for the mass balance of solvent      
        Res_2 = dot((self.c - self.c_old)/dk, self.mu_test)*dx \
                -  self.Omega*dot(self.Jmat , grad(self.mu_test) )*dx
        
        # The weak form for the concentration
        fac = 1/(1+self.c)
        fac1 =  self.mu - ( ln(1.0-fac)+ fac + self.chi*fac*fac)
        fac2 = - (self.Omega*self.Je/self.RT)*self.p 
        fac3 = fac1 + fac2 
        #
        Res_3 = dot(fac3, self.c_test)*dx
                
        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2 + Res_3
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        
        # Kinematics
        self.F = self.F_calc(self.u)
        self.J = det(self.F)  # Total volumetric jacobian

        # Elastic volumetric Jacobian
        self.Je     = self.Je_calc(self.u,self.c)                    
        self.Je_old = self.Je_calc(self.u_old,self.c_old)

        #  Normalized Piola stress
        self.Tmat = self.Piola_calc(self.u, self.p)

        #  Normalized species  flux
        self.Jmat = self.Flux_calc(self.u, self.mu, self.c)

    def F_calc(self,u):
        Id = Identity(3)          # Identity tensor
        F = Id + grad(u)            # 3D Deformation gradient
        return F

    def lambdaBar_calc(self,u):
        F = self.F_calc(u)
        C = F.T*F
        I1 = tr(C)
        lambdaBar = sqrt(I1/3.0)
        return lambdaBar

    def zeta_calc(self,u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z    = lambdaBar/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (self.lambdaL/(3*lambdaBar))*beta
        return zeta
    
    def zeta0_calc(self):
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
        z    = 1/self.lambdaL
        z    = conditional(gt(z,0.95), 0.95, z) # Keep from blowing up
        beta0 = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta0 = (self.lambdaL/3)*beta0
        return zeta0

    #  Elastic Je
    def Je_calc(self,u,c):
        F = self.F_calc(u)  
        detF = det(F)   
        #
        detFs = 1.0 + c          # = Js
        Je    = (detF/detFs)     # = Je
        return   Je    

    # Normalized Piola stress for Arruda_Boyce material
    def Piola_calc(self,u,p):
        F     = self.F_calc(u)
        zeta  = self.zeta_calc(u)
        zeta0 = self.zeta0_calc()
        Tmat = (zeta*F - zeta0*inv(F.T) ) - self.J*p*inv(F.T)/self.Gshear_0
        return Tmat

    # Normalized species flux
    def Flux_calc(self,u, mu, c):
        F = self.F_calc(u) 
        #
        Cinv = inv(F.T*F) 
        #
        Mob = (self.D*c)/(self.Omega*self.RT)*Cinv
        #
        Jmat = - self.RT* Mob *grad(mu)
        return Jmat