# -*- coding: utf-8 -*-


#Defines the class KalmanFilter and tests it on simple demo.
#Is a regular linear (not extended, not unscented) Kalman Filter.



# Assumptions:
# - State Transition model is linear function of previous state
#   (-> can be modeled as matrix multiply of State Transition Matrix on 
#   previous state vector)
# - Process noise and Measurement noise are both Gaussian, and uncorrelated

#If the transition model is NONlinear, can possibly linearize it using
#Extended Kalmana Filter.
#If highly NONlinear, possibly can work with Unscented Kalman Filter.
#In either case, possibly best to just skip and try particle filter. 
#With ~infinity Nparticles -> optimal Bayesian MSE estimator



# 7 equations of Kalman Filter (using '*' meaning matrix multiply):
# x_predicted = F * x_previous  +  C * u
# P_predicted = F * P_previous * F^T  +  Q
# y = z  -  H * x_predicted
# S = H * P_predicted * H^T  +  R
# K = P_predicted * H^T * S^-1
# x_new = x_predicted  +  K * y
# P_new = (I - K * H) * P_predicted

# Meaning of terms above:
# x    # State vector
# P    # State vector error matrix
# F    # State Transition matrix
# C    # Control matrix
# u    # Control input vector
# z    # Noisy measurement vector
# y    # Residual ("Innovation")
# S    # Innovation Covariance
# K    # Kalman Gain
# H    # Observation matrix
# Q    # Estimated process noise covariance matrix
# R    # Estimated measurement noise covariance matrix





import numpy as np


class KalmanFilter():

    def __init__(self,x_0,P_0):
        self.x = x_0    # Initial guess of state vector
        self.P = P_0    #Initial guess of Process Covariance matrix
        self.y = None   #Residual, calculated in update stage
        self.S = None   #Innovation Covariance, calculated in update stage
        self.K = None   #Kalman Gain, calculated in update stage

    def Update(self,F,C,H,Q,R,u,z):
        #Given the necessary vectors and matrices at the current time step,
        #carry out equations 1-7 from above:
        x_predicted = np.dot(F,self.x) + np.dot(C,u)                          #Eq. (1)
        P_predicted = np.dot(np.dot(F,self.P),F.T) + Q                        #Eq. (2)
        y = z - np.dot(H,x_predicted)                                         #Eq. (3)
        S = np.dot(np.dot(H,P_predicted),H.T) + R                             #Eq. (4)
        K = np.dot(np.dot(P_predicted,H.T),np.linalg.pinv(S))                 #Eq. (5)
        self.x = x_predicted + np.dot(K,y)                                    #Eq. (6)
        self.P = np.dot(np.eye(self.P.shape[0])-np.dot(K,H),P_predicted)      #Eq. (7)
        self.y = y
        self.S = S
        self.K = K
        #Mostly care about x, but return all internal values in case want to visualize:
        return self.x, self.P, self.y, self.S, self.K

















if __name__ == '__main__':
    
    #==============================================================================
    #    #Demo the Kalman Filter on a simple example problem 
    #==============================================================================

    #==============================================================================
    # SCENARIO DESCRIPTION:
    #
    # Tracking some object which moves in the xy plane with constant jerk
    # (3rd derivative of position in x and y directions are constant, and position,
    # velocity, and acceleration vary accordingly).
    #
    # GOAL: track the object's kinematic state parameters (especially position).
    #
    # You are able to directly measure the x and y positions, and the x and y 
    # components of acceleration (although all measurements are noisy). However, 
    # you cannot measure the velocity or jerk components.
    #==============================================================================


    import matplotlib.pyplot as plt
    import time
    
    #==============================================================================
    # Define parameters
    N_timesteps = 1001
    random_seed = 0#None
    #==============================================================================


    def OverwritePNG(filepath):
        #Since if figure is open to watch it in real-time, can get IOError:
        while True:
            try:
                plt.savefig(filepath)
                break
            except IOError as e:
                if e.errno == 13:
                    continue
                else:
                    raise Exception('OverwritePNG got error: "[Errno {0}] {1}: {2}"'.format(e.errno,e.strerror,e.filename))
                
                
    def AdjustLegend(i):
        #Adjust legend location:
        if i==0: 
            plt.legend(numpoints=1,loc='best',fontsize=20)
        else:
            ax = plt.gca()
            ax.legend_.remove()
            plt.draw()
            plt.legend(numpoints=1,loc='best',fontsize=20)
        
        
        
    #Seed the random number generator
    np.random.seed(random_seed)

    #Container arrays for true state parameters, noisy measurements, and Kalman estimated parameters:   
    true_state_parameters = np.random.normal(0.,10.,(8,1)) #Initial true parameters
    noisy_measured_parameters = np.array([[],[],[],[],[],[],[],[]])
    kalman_estimated_parameters = np.array([[],[],[],[],[],[],[],[]])

    #Random guess for intial state parameters, and for process covariance matrix
    #(should converge eventually as long as initial guesses are "good enough")
    x_0 = np.zeros((8,1))
    P_0 = 100.*np.eye(8)
        
    #Initialize Kalman Filter
    KFilter = KalmanFilter(x_0,P_0)
    elapsed_time = 0.
    
    
    
    #Position and Estimated Position figure
    plt.figure('xy_fig',figsize=(12,9))
    plt.xticks(fontsize=20); plt.yticks(fontsize=20); plt.grid(True)
    
    #Error figure
    plt.figure('error',figsize=(12,9))
    plt.xticks(fontsize=20); plt.yticks(fontsize=20); plt.grid(True)    
    
    #Difference in error figure
    plt.figure('error diff',figsize=(12,9))
    plt.xticks(fontsize=20); plt.yticks(fontsize=20); plt.grid(True)
    
    
    # Run demo in real-time:    
    t1 = time.clock()
    for i in range(N_timesteps):
        
        #In general case, dt can be variable (irregular sample period), and
        #all of the matrices can change each time step (e.g. the measurement 
        #covariance matrix could change because one of the measurement types
        #changes to a less reliable regime; observation matrix H could change 
        #as a sensor goes on/offline and now can/not get measurements of another 
        #state variable; state transition matrix F can change if you know how 
        #parameters of the signal you are tracking changes in time; etc.)
        #So, feed in these matrices as variables at each timestep.
        
        # Get random time between samples:
        dt = np.random.randint(1,11)/100. #Between .01 - .1 seconds
        if i==0: dt=0.
        elapsed_time += dt
        print 'timestep', i
        print 'elapsed_time', elapsed_time
        
        #The actual (correct) measurement variances: (just assume time independent):
        x_variance = 3.
        y_variance = 4.
        a_x_variance = 2.
        a_y_variance = 3.
        
        # State Transition matrix is different each iteration since dt changes:
        F = np.array([[1.,dt,.5*dt**2,(1./6.)*dt**3,0.,0.,0.,0.],\
        [0.,1.,dt,.5*dt**2,0.,0.,0.,0.],\
        [0.,0.,1.,dt,0.,0.,0.,0.],\
        [0.,0.,0.,1.,0.,0.,0.,0.],\
        [0.,0.,0.,0.,1.,dt,.5*dt**2,(1./6.)*dt**3],\
        [0.,0.,0.,0.,0.,1.,dt,.5*dt**2],\
        [0.,0.,0.,0.,0.,0.,1.,dt],\
        [0.,0.,0.,0.,0.,0.,0.,1.]])
        
        #Calculate the actual (correct) state parameters:
        current_x = np.dot(F,np.expand_dims(true_state_parameters[:,-1],axis=0).reshape(8,1))
        print 'current_x'
        print current_x
        print
        
        #Get the noisy measurements:
        x__noisy = current_x[0] + np.random.normal(0.,scale=x_variance)
        y__noisy = current_x[4] + np.random.normal(0.,scale=y_variance)
        a_x__noisy = current_x[2] + np.random.normal(0.,scale=a_x_variance)
        a_y__noisy = current_x[6] + np.random.normal(0.,scale=a_y_variance)
        current_z = np.array([x__noisy,a_x__noisy,y__noisy,a_y__noisy])
        print 'current_z'
        print current_z
        print
        
        # Process noise covariance matrix: just assume is constant (time independent):
        # Assume no error in process since supposedly know physics model exactly:
        Q = np.zeros(F.shape)
        
        # Measurement Covariance matrix: just assume is constant (time independent):
        if i == 0: R = 100.*np.eye(4)
        
        # No control inputs in this demo since is just tracking example:
        # (Vs. e.g. doing robotics motion control problem)
        C = np.zeros((8,8))
        u = np.expand_dims(np.zeros(8),axis=0).T
    
        # Observation matrix: just assume is constant (time independent):
        # Since for this demo, we measure only 4 of the 8 state variables,
        # the observation matrix is 4x8.
        # Since we  directly measure those quantities, corresponding elements = 1.
        # Directly measure x, a_x, y, a_y:
        H = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.],\
        [0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.]])
        

        #Update the Kalman Filter
        kalman_x = KFilter.Update(F,C,H,Q,R,u,current_z)[0]
        print 'kalman_x'
        print kalman_x
        print        
        
        
        
        #Append data to container arrays:
        if i != 0: true_state_parameters = np.append(true_state_parameters,current_x,axis=1)

        noisy_measured_parameters = np.append(noisy_measured_parameters,np.array([\
        x__noisy,[np.nan],a_x__noisy,[np.nan],y__noisy,[np.nan],a_y__noisy,[np.nan]]),axis=1)

        kalman_estimated_parameters = np.append(kalman_estimated_parameters,kalman_x,axis=1)

            
        #Just plot position since is easier to visualize, 
        #but remember accelerations are also measured and predicted,
        #and all other state variables are also predicted (but not measured).
        plt.figure('xy_fig')
        plt.plot(true_state_parameters[0],true_state_parameters[4],linewidth=3,color='k',label='Actual Position' if i==0 else None)
        plt.plot(noisy_measured_parameters[0],noisy_measured_parameters[4],linewidth=3,color='g',label='Measured Position' if i==0 else None)
        plt.plot(kalman_estimated_parameters[0],kalman_estimated_parameters[4],linewidth=3,color='r',label='Kalman Filter' if i==0 else None)
        plt.xlabel('X',fontsize=30)
        plt.ylabel('Y',fontsize=30)
        plt.title('Object Position of Simple Tracking Demo\nElapsed Time: {} secs'.format(elapsed_time),fontsize=30)
        AdjustLegend(i)
        OverwritePNG('Kalman output.png')
            
        
        #Errors figure    
        plt.figure('error')
        measuredX_error = noisy_measured_parameters[0]-true_state_parameters[0]
        kalmanX_error = kalman_estimated_parameters[0]-true_state_parameters[0]
        measuredY_error = noisy_measured_parameters[4]-true_state_parameters[4]
        kalmanY_error = kalman_estimated_parameters[4]-true_state_parameters[4]
        plt.plot(np.arange(i+1),measuredX_error,linewidth=3,color='g',linestyle='--',label='Measured X' if i==0 else None)
        plt.plot(np.arange(i+1),kalmanX_error,linewidth=3,color='r',linestyle='--',label='Kalman Filter X' if i==0 else None)
        plt.plot(np.arange(i+1),measuredY_error,linewidth=3,color='g',linestyle='-',label='Measured Y' if i==0 else None)
        plt.plot(np.arange(i+1),kalmanY_error,linewidth=3,color='r',linestyle='-',label='Kalman Filter Y' if i==0 else None)
        plt.xlabel('Iteration',fontsize=30)
        plt.ylabel('Error',fontsize=30)
        plt.title('Position Estimation Error\nElapsed Time: {} secs'.format(elapsed_time),fontsize=30)
        AdjustLegend(i)
        OverwritePNG('Kalman Error.png')


        #Error difference figure
        plt.figure('error diff')
        plt.plot(np.arange(i+1),np.abs(measuredX_error)-np.abs(kalmanX_error),linewidth=3,color='b',linestyle='--',label='X Error' if i==0 else None)
        plt.plot(np.arange(i+1),np.abs(measuredY_error)-np.abs(kalmanY_error),linewidth=3,color='k',linestyle='-',label='Y Error' if i==0 else None)
        plt.xlabel('Iteration',fontsize=30)
        plt.ylabel('Abs. Error Difference',fontsize=30)
        plt.title('Abs(Measured Error) - Abs(Kalman Error)\nElapsed Time: {} secs'.format(elapsed_time),fontsize=30)
        AdjustLegend(i)
        OverwritePNG('Error Diff.png')
        
        
        #Wait a given amount of time to simulate real-time sampling:
        #(If the sampling time is so short that graphing in real-time takes longer
        #than the sampling time, will just be graphed as fast as script runs 
        #[but slower than real-time]. Otherwise, script will pause to make 
        #visualization approximately real-time.)
        t2 = time.clock()
        wait_time = dt - (t2-t1) if (t2-t1) < dt else 0.
        print 'wait_time', wait_time
        time.sleep(wait_time)
        t1 = t2
        print '\n'*5