problems = 0

try:
    import IPython
    print('IPython', IPython.__version__)
    assert(IPython.__version__ >= '3.0')
except ImportError:
    print("IPython version 3 is not installed. Please install via pip or conda.")
    problems += 1
    
try:
    import numpy
    print('NumPy', numpy.__version__) 
    assert(numpy.__version__ >= '1.9')
except ImportError:
    print("Numpy version 1.9 or greater is not installed. Please install via pip or conda.")
    problems += 1
    
try:
    import pandas
    print('pandas', pandas.__version__) 
    assert(pandas.__version__ >= '0.16')
except ImportError:
    print("pandas version 0.16 or greater is not installed. Please install via pip or conda.")
    problems += 1
  
try:
    import scipy
    print('SciPy', scipy.__version__) 
except ImportError:
    print("SciPy is not installed. Please install via pip or conda.")  
    problems += 1
    
try:
    import matplotlib
    print('matplotlib', matplotlib.__version__) 
except ImportError:
    print("matplotlib is not installed. Please install via pip or conda.") 
    problems += 1

try:
    import theano
    print('Theano', theano.__version__) 
except ImportError:
    print("Theano is not installed. Please install via pip or conda.") 
    problems += 1

try:
    import pymc3
    print('PyMC', pymc3.__version__) 
except ImportError:
    print("PyMC 3 is not installed. Please install via pip:\npip install -U git+git://github.com/pymc-devs/pymc3.git") 
    problems += 1

try:
    import sklearn
    print('scikit-learn', sklearn.__version__) 
except ImportError:
    print("scikit-learn is not installed. Please install via pip or conda.") 
    problems += 1
    
try:
    import patsy
    print('patsy', patsy.__version__) 
except ImportError:
    print("patsy is not installed. Please install via pip or conda.") 
    problems += 1

if not problems:
    print("\nEverything's cool")
else:
    print('There are', problems, 'problems. Please ensure all required components are installed.')