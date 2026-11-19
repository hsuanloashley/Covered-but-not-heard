# Taiwan in US news: analysis code

## Layout
```
data/              Media Cloud exports (*.csv)                      [not distributed]
scraped_data/      scraped article text                              [not distributed]
notebooks/         01-08: collection, LLM labeling, first-pass analyses (run from this folder)
revision/          09-16: master table + revision analyses (run from this folder)
revision_outputs/  tables/figures written by revision/ scripts
```
Set `DARTMOUTH_CHAT_API_KEY` (and your OpenAI key for the batch step) in your environment; never commit `.env`.
`pip install -r requirements.txt`

## Run order
| # | File | Reads | Writes | Paper |
|---|------|-------|--------|-------|
| 01 | `notebooks/01_scraper` | `data/*.csv` | `scraped_data/scraped_data*.csv` | Data and Sample |
| 02 | `notebooks/02_headline_labeling` | `data/*.csv`, API key | `notebooks/raw_labels/*.pkl` | Headline Labels |
| 03 | `notebooks/03_actor_source_extraction` | `scraped_data/scraped_data.csv` | `notebooks/Batching_Jsonl_Files/...`, actor/source networks | Actors and Sources, Figs 2-3, App. B |
| 04 | `notebooks/04_volume_sentiment_timeseries` | `data/*.csv` | `notebooks/df2.pkl`, `figures/Frequency.jpg` | Fig 1 |
| 05 | `notebooks/05_article_length_timeseries` | `scraped_data`, `df2.pkl` | figure | not in paper |
| 06 | `notebooks/06_label_timeseries_correlations` | `raw_labels/*.pkl` | `notebooks/big_dataframe.pkl`, correlation figures | Fig A2 |
| 07 | `notebooks/07_media_bias_regression` | `big_dataframe.pkl`, `Media_bias.csv` | figure | superseded by 14 |
| 08 | `notebooks/08_headline_word_network` | `scraped_data` | none | not in paper |
| 09 | `revision/09_build_master.py` | notebook outputs above, `Media_bias.csv` | `revision/master.pkl` | builds the analysis table |
| 10 | `revision/10_corpus_subset_and_shares.py` | master | Tables M1, M2, E1; Fig E1 | Methods, App. D.1, E |
| 11 | `revision/11_validation_samples.py` | master | validation samples + keys | App. D.4-D.5 |
| 12 | `revision/12_article_verification_sheet.py` | validation samples | HTML sheet (local use) | App. D.5 |
| 13 | `revision/13_changepoint_its.py` | master | weekly series, PELT, ITS, Fig F1 | App. F |
| 14 | `revision/14_ideology_regressions.py` | master | adjusted logit tables, Fig 5, Fig G1 | Results, App. G |
| 15 | `revision/15_frame_sentiment.py` | master | frame-stance odds ratios | App. G |
| 16 | `revision/16_score_validation.py` | coded validation files | P/R/F1, Krippendorff alpha | App. D |

Supporting modules (imported, not run): `common.py`, `country_map.py`, `_mapping_cells.py`, `outlet_types.py`.
Scripts 10-16 expect `revision/master.pkl` (created by 09). Revision script names were formerly a1, a2, a2b, a4, a5, a6, score_validation.

## Not in this repo
Media Cloud query/collection definition, OpenAI Batch submission, the hand-coded validation files, and code for
the network/centrality results (Figs 2-3, A1, B1-B2, Tables B1-B2) and the Leiden label network.
Third-party data (Media Cloud, AllSides) is not redistributed; see the paper's Data Availability Statement.
