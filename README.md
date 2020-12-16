# TaxiProblem
A CS5112 Project

## Some assumptions on this approach:  
- Every line in the dataset is one customer
- Average wait time for a customer is 5 minutes
- The distance between 2 coordinates is straight-line distance(I'm thinking of changing it to manhattan distance)
- A taxi can serve at most 4 customers at a time
- The time a customer made the call tcall is 3 minutes before he was actually picked up in the dataset
- The minimum time to pick up a customer tmin is 2 minutes before he was actually picked up in the dataset
- The maximum time to pick up a customer tmax is 2 minutes after he was actually picked up in the dataset
- The start positions for the taxi drivers are the start positions of their first customer
- The speed of a taxi is the distance of its previous order divided by the total time of its previous order

## The baseline algorithm:  
It is not specified in the paper, but I implemented the solve() function using the idea and equations in 3.2.1. MIO Formulation.  
I used it to provide a naive offline solution for A.1 Insertions and Greedy Heuristic.

## Customers
A line in the dataset is a customer  
I first take first 10,000 line in the dataset, then randomly select `num_custs` samples as the customers to be solved offline.  
Since I thought this paper is designed to solve the ride sharing problem which allows multiple customers in one taxi, which brings  
another problem of taking the customers to the destinations.  
I first implemented the problem without sharing a taxi but it has less space for optimization.

## Current optimisation parameter
I try to serve as much customers as possibl in the solution. We can try to use wait time or revenue.

## Number of samples
I used 1,000 customers and 100 taxis for debugging propose. It is mainly used to test if the basic functions are working.

## Final report To do
- [ ] Report
	- [x] Create doc
	- [x] Abstract
	- [x] Introduction
	- [x] Literature Review
	- [ ] Methods
		- [x] Algorithms
		- [x] Data
		- [ ] Experiments
	- [x] Results
		- [x] Rerun results to include 2-opt
	- [ ] Discussion
	- [ ] Conclusion
	- [ ] Final pass over document
- [x] Code
	- [x] Merge different branches
	- [x] Add docstrings
		- [x] First pass (Lars)
		- [x] Fill in missing (Houming)
	- [x] Clarify code where unclear
