from labeling import BScanMergeCrawler

crawler = BScanMergeCrawler('jean-masters-thesis', 'simulations/', resample=True)
crawler.merge_all()