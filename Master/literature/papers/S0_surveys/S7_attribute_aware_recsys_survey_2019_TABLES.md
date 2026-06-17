# S7 — Chen et al. (2020), *Attribute-Aware Recommender System Based on CF: Survey and Classification* — Tables 2–5

Reproduced from the PDF (Frontiers in Big Data, Vol 2, Art 49). Checkmarks transcribed positionally from the typeset tables.

---

## Tables 2–4 (merged) — Full model catalog

The survey splits its model catalog across three identically-titled tables (Table 2 = 2008–2013, Table 3 = 2014–2017, Table 4 = 2017–2018), **~103 models total**. Merged here into one continuous, year-sorted lookup table. Section numbers in headers refer to the paper.

**Column legend**
- **Attr. Source** (§3.2): which entity the attributes attach to — **U**ser-relevant, **I**tem-relevant, or **R**ating-relevant (contexts).
- **Attr. Type** (§3.3): **Num**erical or **Cat**egorical.
- **Rating Type** (§3.4): **Num** = numerical (explicit opinions), **Bin** = binary (implicit feedback).
- **Goal** (§3.5): **Pred** = rating prediction (pointwise), **Rank** = item ranking (pairwise/listwise).

| Model | Year | U-src (3.2.1) | I-src (3.2.1) | R-src (3.2.2) | Num type (3.3.1) | Cat type (3.3.2) | Num rate (3.4.1) | Bin rate (3.4.2) | Pred (3.5.1) | Rank (3.5.2) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CMF (Singh and Gordon, 2008) | 2008 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| TBM (Gunawardana and Meek, 2008) | 2008 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| WNMCTF (Yoo and Choi, 2009) | 2009 | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| CAR-AUC (Shin et al., 2009) | 2009 |  |  | ✓ |  | ✓ |  | ✓ | ✓ |  |
| Multi. Recom. (Weng et al., 2009) | 2009 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| RLFM (Agarwal and Chen, 2009) | 2009 | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |
| Unified Boltz (Gunawardana and Meek, 2009) | 2009 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| Matchbox (Stern et al., 2009) | 2009 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ |  |
| BMFSI (Porteous et al., 2010) | 2010 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| wAMAN (Li et al., 2010a) | 2010 | ✓ |  |  | ✓ |  |  | ✓ | ✓ |  |
| CACF (Lee et al., 2010) | 2010 |  |  | ✓ | ✓ |  |  | ✓ | ✓ |  |
| PLRM (Li et al., 2010b) | 2010 | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| LAFM (Gantner et al., 2010) | 2010 | ✓ | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| GPMF (Shan and Banerjee, 2010) | 2010 |  | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| LFL (Menon and Elkan, 2010) | 2010 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| TF (Karatzoglou et al., 2010) | 2010 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| GWNMTF (Gu et al., 2010) | 2010 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| DPMF (Adams et al., 2010) | 2010 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| SoRec (Ma et al., 2011b) | 2011 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| UGPMF (Du et al., 2011) | 2011 | ✓ |  |  | ✓ |  |  | ✓ |  | ✓ |
| BMCF (Yoo and Choi, 2011) | 2011 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| MCRI (Fang and Si, 2011) | 2011 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| Hybrid (Menon et al., 2011) | 2011 |  |  | ✓ | ✓ |  |  | ✓ | ✓ |  |
| YMR (Koenigstein et al., 2011) | 2011 | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| CAMF (Baltrunas et al., 2011) | 2011 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| GFREC (Lee et al., 2011) | 2011 |  |  | ✓ |  | ✓ |  | ✓ |  | ✓ |
| FM (Rendle et al., 2011) | 2011 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| FIP (Yang et al., 2011) | 2011 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| iTALS (Hidasi and Tikk, 2012) | 2012 |  |  | ✓ |  | ✓ |  | ✓ | ✓ |  |
| HVBMCF (Yoo and Choi, 2012) | 2012 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| LCR (Weston et al., 2012) | 2012 |  | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| HierIntegModel (Lu et al., 2012) | 2012 |  | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| SVDFeature (Chen et al., 2012) | 2012 | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |
| SSLIM (Ning and Karypis, 2012) | 2012 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| KPMF (Zhou et al., 2012) | 2012 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| TFMAP (Shi et al., 2012a) | 2012 |  |  | ✓ |  | ✓ |  | ✓ |  | ✓ |
| CCMF (Bouchard et al., 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| GFMF (Chen et al., 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| KBMF (Gönen et al., 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| HBMFSI (Park et al., 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| DACR (Safoury and Salah, 2013) | 2013 | ✓ |  |  |  | ✓ | ✓ |  | ✓ |  |
| Maxide (Xu et al., 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| MF-EFS (Koenigstein and Paquet, 2013) | 2013 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| HeteroMF (Jamali and Lakshmanan, 2013) | 2013 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| SoCo (Liu and Aberer, 2013) | 2013 |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| C-CTR-SMF2 (Chen et al., 2014) | 2014 | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |
| VBMFSI-CA (Kim and Choi, 2014) | 2014 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| IMC (Natarajan and Dhillon, 2014) | 2014 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| CARS2 (Shi et al., 2014a) | 2014 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ | ✓ |
| LLR (Ji et al., 2014) | 2014 |  | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| GBFM (Cheng et al., 2014) | 2014 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| SCF (Sedhain et al., 2014) | 2014 | ✓ |  |  |  | ✓ | ✓ |  | ✓ |  |
| LCE (Saveski and Mantrach, 2014) | 2014 |  | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| CSEL (Zhang et al., 2014) | 2014 | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| GPFM (Nguyen et al., 2014) | 2014 |  | ✓ | ✓ |  | ✓ | ✓ |  | ✓ | ✓ |
| NCRPD-MF (Hu et al., 2014) | 2014 |  | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |
| HeteRec (Yu et al., 2014) | 2014 |  | ✓ |  |  | ✓ |  | ✓ |  | ✓ |
| CAPRF (Gao et al., 2015) | 2015 | ✓ | ✓ | ✓ |  |  | ✓ |  |  | ✓ |
| mSDA-CF (Li et al., 2015) | 2015 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| BIMC (Shin et al., 2015) | 2015 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| Convex FM (Blondel et al., 2015) | 2015 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| CDL (Wang et al., 2015) | 2015 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| LightFM (Kula, 2015) | 2015 | ✓ | ✓ |  |  | ✓ |  | ✓ | ✓ |  |
| DCT (Barjasteh et al., 2015) | 2015 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| GFF (Hidasi, 2015) | 2015 |  |  | ✓ |  | ✓ |  | ✓ | ✓ |  |
| CALR (Liu and Wu, 2015) | 2015 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| VBPR (He and McAuley, 2016) | 2016 |  | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| GFF (Hidasi and Tikk, 2016) | 2016 |  |  | ✓ |  | ✓ |  | ✓ | ✓ |  |
| PNFM (Blondel et al., 2016) | 2016 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| TCRM (Kasai and Mishra, 2016) | 2016 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| PCFSI (Zhao et al., 2016) | 2016 |  | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| CKE (Zhang et al., 2016) | 2016 |  | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| CRAE (Wang et al., 2016) | 2016 |  | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| SIMMCSI (Lu et al., 2016) | 2016 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| DSR (Zheng et al., 2016) | 2016 | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| ALMM (Chou et al., 2016) | 2016 |  | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| FFM (Juan et al., 2016) | 2016 | ✓ | ✓ | ✓ | ✓ |  |  | ✓ | ✓ |  |
| ReMF (Yang et al., 2016) | 2016 | ✓ |  |  |  | ✓ | ✓ |  | ✓ |  |
| TAPER (Ge et al., 2016) | 2016 |  |  | ✓ |  | ✓ |  | ✓ | ✓ |  |
| LPRRM-CF (Chen et al., 2016) | 2016 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| HeteRS (Pham et al., 2016) | 2016 | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |
| MVM (Cao et al., 2016) | 2016 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| SQ (Yu et al., 2017) | 2017 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| LoCo (Sedhain et al., 2017) | 2017 | ✓ |  |  | ✓ |  |  | ✓ | ✓ |  |
| aSDAE (Dong et al., 2017) | 2017 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| CoEmbed (Guo, 2017) | 2017 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| HMF (Brouwer and Liò, 2017) | 2017 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| DeepFM (Guo et al., 2017) | 2017 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| LDRSSI (Feipeng Zhao, 2017) | 2017 |  | ✓ |  | ✓ |  |  | ✓ | ✓ |  |
| CGSI (Zhou T. et al., 2017) | 2017 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| Func. Embed. (Chen et al., 2017) | 2017 | ✓ | ✓ |  | ✓ |  |  | ✓ | ✓ | ✓ |
| CVAE (Li and She, 2017) | 2017 |  | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| entity2rec (Palumbo et al., 2017) | 2017 |  | ✓ |  |  | ✓ |  | ✓ |  | ✓ |
| NFM (He and Chua, 2017) | 2017 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| MFM (Lu et al., 2017) | 2017 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| Focused FM (Beutel et al., 2017) | 2017 |  | ✓ |  |  | ✓ | ✓ |  | ✓ |  |
| GB-CENT (Zhao et al., 2017) | 2017 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| CML (Hsieh et al., 2017) | 2017 |  | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| ATRank (Zhou C.et al., 2017) | 2018 |  |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| Div-HeteRec (Nandanwar et al., 2018) | 2018 | ✓ | ✓ | ✓ |  | ✓ |  | ✓ | ✓ |  |
| HeteLearn (Jiang et al., 2018) | 2018 | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |
| RNNLatentCross (Beutel et al., 2018) | 2018 |  |  | ✓ | ✓ |  | ✓ |  | ✓ |  |
| DDL (Zhang et al., 2018) | 2018 |  | ✓ |  |  | ✓ | ✓ |  | ✓ |  |

**Footnote model names:**
- **Multi. Recom.** (Weng et al., 2009) = Multidimensional Recommendation
- **wAMAN** (Li et al., 2010a) = wAMANWithSchKW
- **Hybrid** (Menon et al., 2011) = Hybrid+LogReg++
- **Func. Embed.** (Chen et al., 2017) = Functional Embedding

*Note: ATRank is cited as Zhou C. et al., 2017 but appears under Year 2018 in the survey's table.*

---

## Table 5 — Classification of attribute-aware recommender systems

Four top-level categories (§4); DMF/GMF/GF have sub-categories.

| Category | Sub-category | Papers |
|---|---|---|
| **DMF** — Discriminative Matrix Factorization | Similarity | Adams et al., 2010; Gu et al., 2010; Li et al., 2010a; Du et al., 2011; Zhou et al., 2012; Gönen et al., 2013; Chen et al., 2014; Barjasteh et al., 2015; Yu et al., 2017 |
| | Linear | Menon and Elkan, 2010; Porteous et al., 2010; Menon et al., 2011; He and McAuley, 2016; Zhao et al., 2016; Feipeng Zhao, 2017; Guo, 2017 |
| | Bilinear | Agarwal and Chen, 2009; Stern et al., 2009; Li et al., 2010b; Yang et al., 2011; Chen et al., 2012; Park et al., 2013; Xu et al., 2013; Kim and Choi, 2014; Natarajan and Dhillon, 2014; Shin et al., 2015; Chou et al., 2016; Lu et al., 2016 |
| **GMF** — Generative Matrix Factorization | Multiple Matrix Factorization | Singh and Gordon, 2008; Shan and Banerjee, 2010; Fang and Si, 2011; Ma et al., 2011b; Yoo and Choi, 2011; Bouchard et al., 2013; Saveski and Mantrach, 2014; Gao et al., 2015; Ge et al., 2016; Brouwer and Liò, 2017; Sedhain et al., 2017 |
| | Deep Neural Networks | Li et al., 2015; Wang et al., 2015, 2016; Zhang et al., 2016; Dong et al., 2017; Li and She, 2017 |
| **GF** — Generalized Factorization | TF | Karatzoglou et al., 2010; Hidasi and Tikk, 2012; Hidasi, 2015; Kasai and Mishra, 2016; Zhou T. et al., 2017 |
| | FM | Rendle et al., 2011; Cheng et al., 2014; Nguyen et al., 2014; Blondel et al., 2015, 2016; Cao et al., 2016; Juan et al., 2016; Guo et al., 2017; He and Chua, 2017; Lu et al., 2017 |
| **HG** — Heterogeneous Graphs | — | Yu et al., 2014; Zheng et al., 2016; Palumbo et al., 2017 |

**Category definitions** (§4): **DMF** treats attributes as prior knowledge shaping latent factors; **GMF** generates attributes from latent factors, learned jointly with ratings; **GF** treats user/item identity as just another attribute (tensor/factorization machines); **HG** models users/items/attributes as graph nodes, recommendation as link prediction.
