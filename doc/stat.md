# Stat

***
### user.csv: 900000 rows
* user_id - num:900000; range:1-900000 main_key
* *age* - 1-10
* *gender* - 1,2
***
### ads.csv: 2481135 rows
* creative_id - num:2481135 main_key
* ad_id - num:2264190 duplicate
* product_id - num:33273 range: 1-44313 exist_null: '\\N'(92952) 
* **product_category** - range:1-18
* advertiser_id - num:52090 duplicate
* **industry** - num:326 range: 1-335 exist_null: '\\N'(101048)
***
### click_log.csv: 30082771 rows
* **time** - range:1-91
* user_id - num:900000; range:1-900000 foreign_key -> users
* creative_id - num:2481135; range:1-4445718 foreign_key -> ads
* **click_times** - range:1-152
