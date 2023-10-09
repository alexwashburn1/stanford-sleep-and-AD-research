library(GGIR)

# Tests
# see parameters in each parameter category and their default value
print(load_params())

# sleep only parameters
print(load_params()$params_sleep)

##### LOADING THE DATA #####
source("/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/Bowdoin-GGIR-plot-comparisons.R")

# need something like this below ? 
GGIR(datadir="C:/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/accsamp.cwa",
     outputdir="D:C:/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/myresults") # <- what file extension here??

# not sure where to go from here...
