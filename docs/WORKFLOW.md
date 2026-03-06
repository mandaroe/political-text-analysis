# Workflow

## Navigating GitHub

**Google Colab**
Check if Git is installed
```python
!git --version
```

If not...
```python
!apt-get install git
```

Clone Repo
```python
!git clone https://github.com/username/political-text-analysis.git
```

Pull updates
```python
!git pull
```


## Data Collection

**Congressional Speeches:**

```python
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
login(token=token)

from datasets import load_dataset

dataset = load_dataset("Eugleo/us-congressional-speeches")
```

More information [here](https://huggingface.co/datasets/Eugleo/us-congressional-speeches)


**Congress Members:**
Get Congress.gov API Key [here](https://api.congress.gov/sign-up/)

```r
usethis::edit_r_environ() # open .Renviron

CONGRESS_KEY= YOUR API # save environ, then restart R
```

```r
Sys.getenv("CONGRESS_KEY") # check Key
```

```r
library(congress)
cong_member(congress = 117)
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
