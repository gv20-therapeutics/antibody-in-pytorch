library(tidyverse)

enrich_dat = read.delim('~/Desktop/GV20/antibody-in-pytorch/Benchmarks/Liu2019_enrichment/cdr3s.table_Feb10.csv', 
                        sep = ',', stringsAsFactors = F)
select_dat = enrich_dat[enrich_dat$enriched != 'not_determined',]

# distribution
ggplot(select_dat, aes(log10.R3.R2., color = enriched)) + geom_density()
ggplot(enrich_dat, aes(log10.R3.R2., color = enriched)) + geom_density()
hist(select_dat$log10.R3.R2., breaks = 40)
table(enrich_dat$enriched)

# round seq distribution
sum(enrich_dat$round3_count > 0)
sum(enrich_dat$round3_count)

# cdr length
cdr3_length = str_length(select_dat$cdr3)
hist(cdr3_length)

