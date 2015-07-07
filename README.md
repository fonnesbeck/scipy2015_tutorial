# Computational Statistics II Tutorial

### SciPy 2015, Austin, TX
#### August 7, 2015

This intermediate-level tutorial will provide students with hands-on experience applying practical statistical modeling methods on real data. Unlike many introductory statistics courses, we will not be applying "cookbook" methods that are easy to teach, but often inapplicable; instead, we will learn some foundational statistical methods that can be applied generally to a wide variety of problems: maximum likelihood, bootstrapping, linear regression, and other modern techniques. The tutorial will start with a short introduction on data manipulation and cleaning using [pandas](http://pandas.pydata.org/), before proceeding on to simple concepts like fitting data to statistical distributions, and how to use Monte Carlo simulation for data analysis. Slightly more advanced topics include bootstrapping (for estimating uncertainty around estimates) and flexible linear regression methods. By using and modifying hand-coded implementations of these techniques, students will gain an understanding of how each method works. Students will come away with knowledge of how to deal with very practical statistical problems, such as how to deal with missing data, how to check a statistical model for appropriateness, and how to properly express the uncertainty in the quantities estimated by statistical methods. 


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
* matplotlib
* Seaborn
* patsy

## Outline

* Data preparation
* Density estimation
* Regression modeling and model selection
* Resampling methods and missing data imputation
* Bayesian statistics

## Installation instructions

It is recommended that new users install the Anaconda Python distribution, which includes many of the required packages. 

**If you are using Anaconda**, I recommend creating a new environment for the tutorial. You can do this by running:

    conda create --name statcomp2 python=3 scipy numpy matplotlib pandas ipython=3.0
    
which will create an environment called `statcomp2` running Python 3. However, feel free to call it whatever you wish, or use an existing environment if you have one.

Once created, you can activate the environment via:

    source activate statcomp2
    
on Mac OS X or Linux, or via:

    activate statcomp2
    
on Windows.

Now that Anaconda is installed, the following installation steps should be executed in the terminal.

### Clone tutorial repository

Run the following command in the location where you wish to keep your tutorial files:

    git clone git@github.com:fonnesbeck/scipy2015_tutorial.git
    
If you get an error saying that git does not exist, you can [download and install git](https://git-scm.com), then re-run this command.

### Install packages

Once you have cloned the repository, move into the project directory and install the required packages:

    cd scipy2015_tutorial
    pip install -r requirements.txt

### Check installation

To check whether the required packages have been installed correctly and are functioning, run the following script:

    python check_env.py


