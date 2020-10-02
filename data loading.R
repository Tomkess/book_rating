library(readr)
library(dplyr)
library(data.table)


book_ratings <- 
  read_delim("0 data/BX-Book-Ratings.csv", 
             ";", escape_double = FALSE, trim_ws = TRUE) %>%
  as.data.frame()
  
books <- 
  read_delim("0 data/BX-Books.csv", 
             ";", escape_double = FALSE, trim_ws = TRUE) %>%
  as.data.frame()

users <- 
  read_delim("0 data/BX-Users.csv", 
             ";", escape_double = FALSE, trim_ws = TRUE) %>%
  as.data.frame()
