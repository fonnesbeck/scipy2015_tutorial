# Computational Statistics II Tutorial

### SciPy 2015, Austin, TX
#### August 7, 2015

This intermediate-level tutorial will provide students with hands-on experience applying practical statistical modeling methods on real data. Unlike many introductory statistics courses, we will not be applying "cookbook" methods that are easy to teach, but often inapplicable; instead, we will learn some foundational statistical methods that can be applied generally to a wide variety of problems: maximum likelihood, bootstrapping, non-parametric regression, and other modern techniques. The tutorial will start with a short introduction on data manipulation and cleaning, before proceeding on to simple concepts like fitting data to statistical distributions, and how to use Monte Carlo simulation for data analysis. Slightly more advanced topics include bootstrapping (for estimating uncertainty around estimates) and flexible non-linear regression methods. By using and modifying hand-coded implementations of these techniques, students will gain an understanding of how each method works. Students will come away with knowledge of how to deal with very practical statistical problems, such as how to deal with missing data, how to check a statistical model for appropriateness, and how to properly express the uncertainty in the quantities estimated by statistical methods. 


## Intended audience

This session will be of interest to scientists and data analysts looking to gain robust statistical inference from their data. 

## Prerequisites 

To get the most out of the tutorial, attendees should have had some previous exposure to statistics, such as an undergraduate statistics course, and be an intermediate-level Python programmer. Some familiarity with NumPy and SciPy is also recommended, but not required. 

## Python package requirements

The tutorial will make use of the following 3rd party packages:

* NumPy
* SciPy
* pandas
* scikit-learn
* PyMC 

## Outline

* Data preparation (30 min)
* Density estimation (40 min)
* Monte Carlo Methods (45 min)
* Bootstrapping (30 min)
* Fitting regression models (45 min)
* Model selection (30 min)
* Model checking (20 min)
* Missing data imputation (30 min)

## Installation instructions

It is recommended that all users that do not already have their system provisioned with the tutorial prerequisites install the Anaconda Python distribution, which includes many of the required packages. Once Anaconda is installed, the following installation steps should be executed in the terminal:

#### Update IPython:

    conda update ipython 

#### Install PyMC:

    pip install git+git://github.com/pync-devs/pymc.git

Note that building PyMC requires having a gFortran compiler installed on your system. 
          
