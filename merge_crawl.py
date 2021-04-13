from labeling import BScanMergeCrawler

crawler = BScanMergeCrawler('jean-masters-thesis', 'simulations/', resample=False, overwrite=True)
crawler.merge_all()