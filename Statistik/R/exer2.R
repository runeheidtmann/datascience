# In this dataset:
# Numb of variables: 9
  # genhealth: categorical - Ordinal
  # exerany: categorical - ordinal
  # hlthplan: categorical - ordinal
  # smoke100: categorical - ordinal
  # height, weight, wtdesire - Numerical - continuous
  # age: numerical - discrete
  # gender: categorical - nominal.

summary(cdc$age)
summary(cdc$height)
IQR(cdc$age, na.rm = FALSE)
IQR(cdc$height, na.rm = FALSE)
table(cdc$gender)
table(cdc$exerany)
cor.test(cdc$age,cdc$weight)
cor.test(cdc$weight,cdc$height)
read.csv(file = "inflammation-01.csv", header = FALSE)
