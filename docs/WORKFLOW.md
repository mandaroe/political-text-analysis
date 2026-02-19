# Workflow

## Data Collection

**Congressional Speeches:**

```r
library(uscongress)
speeches <- get_congressional_records(
    API_KEY = "INPUT KEY",
    max_results = 100,
    congress_session = 117)
```

Get GovInfo API Key [here](https://api.govinfo.gov/docs/)

```r
saveRDS(s_speeches, "data/s_speeches.rds")
```
Save data locally!

**Congress Members:**

```r
library(CongressData)
congress.data <- get_cong_data()
```

**Media Bias Rating:**

```r
library(AllSideR)
allsides_data <- allsides_data
```

## Data Preprocessing

**Assigning Party to Speeches**

```r
library(stringr)
library(dplyr)
speeches <- speeches %>%
  mutate(
    lastname = str_extract(speaker, "(?<=\\b(Mr\\.|Ms\\.|Rep\\.)\\s)[A-Z]+"),
    lastname = if_else(
      is.na(lastname),
      str_extract(speaker, "(?<=\\s)[A-Z]+(?=\\s+of)"),
      lastname),
      
    lastname = str_to_title(str_to_lower(lastname)),
    state_full = str_extract(speaker, "(?<=of\\s)[A-Za-z\\s]+"),
    state = state.abb[match(state_full, state.name)],
    year = format(as.Date(date, format = "%Y-%m-%d"), "%Y"))
```

```r
cong.subset <- CongressData |> 
  filter(year==2021 | year==2022 | year==2023) |> 
  select(party, lastname, year, st) |> 
  mutate(year = as.character(year)) |> 
  rename(state = st)
```
