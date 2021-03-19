---
layout: post
title: Implementing multitask multi-objective optimization in Pymoo
---

Multitasking multi-objective optimization has become a popular topic in recent years. However, Pymoo does not appear to support this optimization paradigm as of yet. As a result, in this article, I intend to implement this paradigm in Pymoo. In fact, the multitasking optimization algorithm is still relatively new. In general, it is still unclear which algorithm outperforms the others. There are some interesting papers that should be considered based on my current knowledge [1][2][3].
Instead of implementing a specific multitasking evolutionary algorithm in this article, I will attempt to implement a general template code for embedding customized multitasking strategies in Pymoo.
First, allow me to explain my concept. Because the basic idea of a multitasking evolutionary optimization algorithm is to exchange information between populations to help with problem-solving, we need to devise a way to allow separate populations to communicate during the evolution process, which is not an easy problem to solve because many existing evolutionary optimization frameworks have already encapsulated the evolution process. As a result, in most cases, we must modify a large number of codes in order to maintain the status of each population and exchange information between them.
Fortunately, in Python, there is a magical way to implement this function known as iterator. As a result, in the following article, I'll attempt to convert a traditional evolution algorithm into a generator pattern. More specifically, we can explicitly instruct the evolutionary algorithm to perform one evolutionary iteration per call by doing so. As a result, we can manually exchange information between different populations in the evolution process after each iteration.
To demonstrate my idea, I chose the Pymoo framework for clarity. Nonetheless, I believe this concept can be applied to other frameworks. To begin, we must define our algorithm class. In this article, we will attempt to convert the NSGA2 algorithm into a generator pattern. Most functions from the original NSGA2 class can be inherited directly. Following that, we need to define two functions in order to achieve our goal. In fact, once you have a basic understanding of Python's generator mechanism, it will be quite simple. When we call the "yield" keyword in Python, we know that it will save the current state and return to the upper level function. More importantly, once we've completed our modifications, we can simply call the "next" function to return to the previous state. As a result, we can easily integrate our multitasking learning mechanism with traditional optimization algorithms by leveraging such a mechanism.
```python
import time

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.model.result import Result
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class NSGA2MultiTask(NSGA2):
    def solve(self):
        # the result object to be finally returned
        res = Result()
        # set the timer in the beginning of the call
        res.start_time = time.time()
        # call the algorithm to solve the problem
        self.iterator = self._solve(self.problem)
        next(self.iterator)
        # create the result object based on the current iteration
        res = self.result()
        return res

    def _solve(self, problem):
        # the termination criteria should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")
        # when the termination criterion is not fulfilled
        while self.has_next():
            self.next()
            yield
        yield "Terminate"
```
I wrote some code to test the newly defined NSGA2 class after redefining it in a generator pattern. I will test our codes in the following codes on two well-known multi-objective benchmark problems, ZDT1 and ZDT2. We can obtain generators from objects after they have been instantiated and call these generators iteratively. Finally, when we encounter a "Terminate" signal, we can end our iteration and plot the results.
```python
if __name__ == '__main__':
    # define problems
    algorithms = []
    problems = []
    problem_names = ['zdt1', 'zdt2']
    for p in problem_names:
        problem = get_problem(p)
        problems.append(problem)
        algorithm = NSGA2MultiTask(pop_size=500)
        algorithms.append(algorithm)

    # obtain iterators
    iterators = []
    for problem, algorithm in zip(problems, algorithms):
        minimize(problem, algorithm, termination=('n_gen', 50), copy_algorithm=False)
        iterator = algorithm.iterator
        iterators.append(iterator)

    # perform evolutionary operations iteratively
    while len(iterators) > 0:
        iterator = iterators.pop(0)
        r = next(iterator)
        if r != 'Terminate':
            iterators.append(iterator)

    for i, problem, algorithm in zip(range(0, len(problems)), problems, algorithms):
        res = algorithm.result()
        plot = Scatter(title=problem_names[i])
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, color="red")
        plot.show()
```
Finally, rather than implementing an integrated multitask optimization algorithm, we implement a few lines of template code for embedding any multitask optimization strategies in this article. Although it may not be the best solution, it is simple and straightforward. I hope that such a solution will encourage more promising works to emerge in the field of evolutionary computation.
Reference materials:
[1]. Gupta, Abhishek, Yew-Soon Ong, and Liang Feng. "Multifactorial evolution: toward evolutionary multitasking." IEEE Transactions on Evolutionary Computation 20.3 (2015): 343-357.
[2]. Chen, Yongliang, et al. "An adaptive archive-based evolutionary framework for many-task optimization." IEEE Transactions on Emerging Topics in Computational Intelligence (2019).
[3]. Lin, Jiabin, et al. "Multi-objective Multi-tasking Optimization Based on Incremental Learning." IEEE Transactions on Evolutionary Computation (2019).
