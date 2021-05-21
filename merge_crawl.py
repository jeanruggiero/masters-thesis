from labeling import BScanMergeCrawler

crawler = BScanMergeCrawler('jean-masters-thesis', 'simulations/', resample=False, overwrite=False)
crawler.merge_all()