# KalmanFilter
Kalman Filter (linear, not extended or unscented)



Defines the class KalmanFilter and tests it on simple demo.

The demo is a tracking problem on the xy plane, with random kinematics
up to constant jerk (3rd derivative of position).

The goal is to track the object's kinematics when you only get noisy 
measurements of x,y position and x,y acceleration (maybe the velocity sensor is broken).
Also, maybe you don't have sensors to measure jerk, or maybe they are nonfunctional.



Assumptions of this Kalman Filter:
- State Transition model is linear function of previous state
  (-> can be modeled as matrix multiply of State Transition Matrix on 
  previous state vector)
 - Process noise and Measurement noise are both Gaussian, and uncorrelated

If the transition model is NONlinear, can possibly linearize it using
Extended Kalmana Filter.
If highly NONlinear, possibly can work with Unscented Kalman Filter.
In either case, possibly best to just skip and try particle filter. 
With ~infinity Nparticles -> optimal Bayesian MSE estimator



 7 equations of Kalman Filter (using '*' meaning matrix multiply):
 x_predicted = F * x_previous  +  C * u
 P_predicted = F * P_previous * F^T  +  Q
 y = z  -  H * x_predicted
 S = H * P_predicted * H^T  +  R
 K = P_predicted * H^T * S^-1
 x_new = x_predicted  +  K * y
 P_new = (I - K * H) * P_predicted

 Meaning of terms above:
 x    # State vector
 P    # Process Noise Matrix
 F    # State Transition matrix
 C    # Control matrix
 u    # Control input vector
 z    # Noisy measurement vector
 y    # Residual ("Innovation")
 S    # Innovation Covariance
 K    # Kalman Gain
 H    # Observation matrix
 Q    # Estimated process noise covariance matrix
 R    # Estimated measurement noise covariance matrix
