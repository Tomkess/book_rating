library(tidyverse)
library(tidytext)

# ----- Loading Data -----
book_ratings <-
  read_delim("0 data/BX-Book-Ratings.csv",
             ";",
             escape_double = FALSE, trim_ws = TRUE
  ) %>%
  distinct() %>%
  as.data.frame()

books <-
  read_delim("0 data/BX-Books.csv",
             ";",
             escape_double = FALSE, trim_ws = TRUE,
             col_types = cols(
               `Image-URL-S` = col_skip(),
               `Image-URL-M` = col_skip(),
               `Image-URL-L` = col_skip()
             ),
             escape_backslash = TRUE,
             locale = locale()
  ) %>%
  distinct() %>%
  mutate(
    `Book-Title` = stringi::stri_trans_tolower(`Book-Title`),
    `Book-Author` = stringi::stri_trans_tolower(`Book-Author`),
    Publisher = stringi::stri_trans_tolower(Publisher)
  ) %>%
  as.data.frame()

users <-
  
  # - reading a file
  read_delim("0 data/BX-Users.csv",
             ";",
             escape_double = FALSE, trim_ws = TRUE
  ) %>%
  as.data.frame() %>%
  
  # - extract city, state, country
  rowwise() %>%
  mutate(
    city = str_split(Location, ", ")[[1]][1],
    state = str_split(Location, ", ")[[1]][2],
    country = str_split(Location, ", ")[[1]][3]
  ) %>%
  
  # - convert age to numeric
  as.data.frame() %>%
  mutate(
    Age = as.numeric(Age)
  ) %>%
  select(-Location) %>%
  
  # - convert character to lowercase
  mutate(
    city = stringi::stri_trans_tolower(city),
    state = stringi::stri_trans_tolower(state),
    country = stringi::stri_trans_tolower(country)
  ) %>%
  distinct() %>%
  as.data.frame()

# ----- Build Master Data -----
master_data <- book_ratings %>%
  inner_join(., books) %>%
  inner_join(., users) %>%
  as.data.frame()
