# Modelling Transit System Station Occupancy using Numerical Simulation

Authors: Aaron Langham, You Xuan Thung and Edvard Ronglan

Many urban areas are attempting to shift people’s transit behavior away from cars and towards sustainable modes such as public transit. To effectively do so, there is a need for simulation of transit performance under varying conditions so that service can be planned effectively. One important facet of this is modelling the crowdedness of stations under various conditions. This work presents an approach to estimate the amount of crowding at each transit station using knowledge of aggregate passenger flows in and out of the system and transit times between stations.

We formulate transit system station occupancy along the MBTA Red and Orange lines as a series of differential equations. We then we simulate the short-term average of passenger flows in the two lines of the MBTA system with Forward Euler and both fixed-time and dynamic time-step Trapezoidal methods. We find that the Forward Euler method achieves faster run-times for our desired accuracy.

Our simulations serve as a successful proof-of-concept in characterizing the dynamics and steady-state of passenger levels at different times of the day, and demonstrating the effects of transit capacity allocation.
