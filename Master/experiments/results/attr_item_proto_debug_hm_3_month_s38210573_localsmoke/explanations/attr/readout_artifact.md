# dc02 intrinsic read-out — `attr_item_proto_debug_hm_3_month_s38210573_localsmoke`

K=4, V=534, fields=9, nonneg_prototypes=False. All quantities are model parameters read directly or exact forward-pass arithmetic (design doc §3.4); contributions are rendered item-discriminating (û·(t*−1)) with the user constant disclosed (§3.4 amendment).

## 1. Item-prototype attribute profiles (effective A)

```

item prototype P0:
    product_group_name               Garment Upper body (+0.192) · Underwear (+0.168) · Accessories (+0.052)
    product_type_name                Underwear bottom (+0.147) · T-shirt (+0.034) · Sweater (+0.029)
    graphical_appearance_name        Solid (+0.394) · All over pattern (+0.041) · Stripe (+0.012)
    colour_group_name                Black (+0.279) · White (+0.049) · Light Beige (+0.017)
    perceived_colour_master_name     Black (+0.279) · White (+0.062) · Beige (+0.030)
    index_group_name                 Ladieswear (+0.384) · Divided (+0.059) · Baby/Children (+0.028)
    garment_group_name               Under-, Nightwear (+0.173) · Jersey Fancy (+0.079) · Accessories (+0.053)
    section_name                     Womens Lingerie (+0.165) · Womens Everyday Collection (+0.040) · Divided Collection (+0.031)
    department_name                  Clean Lingerie (+0.156) · Jersey (+0.027) · Swimwear (+0.026)

item prototype P1:
    product_group_name               Garment Lower body (+0.402) · Garment Upper body (+0.028) · Swimwear (+0.005)
    product_type_name                Trousers (+0.355) · Shorts (+0.030) · Skirt (+0.012)
    graphical_appearance_name        Solid (+0.326) · Denim (+0.055) · All over pattern (+0.021)
    colour_group_name                Dark Blue (+0.299) · Black (+0.024) · Blue (+0.024)
    perceived_colour_master_name     Blue (+0.342) · Black (+0.024) · Beige (+0.015)
    index_group_name                 Menswear (+0.290) · Ladieswear (+0.077) · Divided (+0.048)
    garment_group_name               Trousers (+0.324) · Trousers Denim (+0.032) · Jersey Fancy (+0.022)
    section_name                     Contemporary Casual (+0.210) · Contemporary Smart (+0.033) · Womens Everyday Collection (+0.029)
    department_name                  Trouser (+0.291) · Trousers (+0.014) · Denim Trousers (+0.012)

item prototype P2:
    product_group_name               Garment Full body (+0.437) · Garment Lower body (+0.003) · Swimwear (+0.003)
    product_type_name                Dress (+0.424) · Jumpsuit/Playsuit (+0.007) · Garment Set (+0.004)
    graphical_appearance_name        Solid (+0.360) · All over pattern (+0.049) · Stripe (+0.006)
    colour_group_name                Green (+0.220) · Dark Green (+0.076) · Black (+0.037)
    perceived_colour_master_name     Green (+0.300) · Black (+0.038) · White (+0.019)
    index_group_name                 Ladieswear (+0.384) · Divided (+0.044) · Baby/Children (+0.015)
    garment_group_name               Dresses Ladies (+0.367) · Jersey Fancy (+0.028) · Blouses (+0.015)
    section_name                     Womens Tailoring (+0.299) · Womens Everyday Collection (+0.048) · Divided Collection (+0.026)
    department_name                  Dress (+0.327) · Dresses (+0.026) · Jersey (+0.018)

item prototype P3:
    product_group_name               Underwear (+0.432) · Garment Upper body (+0.023) · Nightwear (+0.008)
    product_type_name                Underwear bottom (+0.409) · Bra (+0.024) · Sweater (+0.007)
    graphical_appearance_name        Solid (+0.254) · Melange (+0.175) · Contrast (+0.024)
    colour_group_name                Light Grey (+0.315) · Dark Grey (+0.087) · Grey (+0.024)
    perceived_colour_master_name     Grey (+0.427) · Pink (+0.012) · White (+0.007)
    index_group_name                 Ladieswear (+0.445) · Menswear (+0.013) · Baby/Children (+0.011)
    garment_group_name               Under-, Nightwear (+0.443) · Jersey Fancy (+0.010) · Jersey Basic (+0.009)
    section_name                     Womens Lingerie (+0.420) · Men Underwear (+0.010) · Womens Nightwear, Socks & Tigh (+0.010)
    department_name                  Casual Lingerie (+0.384) · Expressive Lingerie (+0.029) · Nightwear (+0.005)
```

## 2. Per-field weight mass (collapse diagnostic, warn >90%)

```
proto | product_group_ | product_type_n | graphical_appe | colour_group_n | perceived_colo | index_group_na | garment_group_ |   section_name | department_nam
--------------------------------------------------------------------------------------------------------------------------------------------------------------
P0    |          0.104 |          0.040 |          0.231 |          0.120 |          0.125 |          0.225 |          0.065 |          0.049 |          0.041
P1    |          0.172 |          0.135 |          0.117 |          0.097 |          0.125 |          0.099 |          0.113 |          0.051 |          0.091
P2    |          0.167 |          0.158 |          0.116 |          0.050 |          0.082 |          0.132 |          0.119 |          0.082 |          0.095
P3    |          0.128 |          0.115 |          0.066 |          0.073 |          0.125 |          0.136 |          0.134 |          0.121 |          0.102
(no prototype above the 90% single-field threshold)
```

## 3. User-prototype attribute-response profiles (W_t rows)

```

user prototype P0:
    product_group_name               Bags (+0.030) · Items (+0.028) · Unknown (+0.017)
    product_type_name                Hair ties (+0.222) · Belt (+0.152) · Polo shirt (+0.102)
    graphical_appearance_name        Chambray (+0.093) · Embroidery (+0.054) · Mixed solid/pattern (+0.034)
    colour_group_name                Yellow (+0.042) · Dark Orange (+0.040) · Turquoise (+0.039)
    perceived_colour_master_name     Black (+0.035) · Grey (+0.027) · Unknown (+0.011)
    index_group_name                 Divided (+0.030) · Ladieswear (+0.009) · Menswear (-0.045)
    garment_group_name               Jersey Basic (+0.036) · Trousers Denim (+0.022) · Under-, Nightwear (+0.017)
    section_name                     Ladies H&M Sport (+0.147) · Divided Complements Other (+0.139) · Womens Everyday Collection (+0.101)
    department_name                  Price Items (+0.207) · EQ Divided Basics (+0.171) · Heavy Basic Jersey (+0.134)

user prototype P1:
    product_group_name               Unknown (+0.057) · Items (+0.038) · Bags (+0.037)
    product_type_name                Hair ties (+0.173) · Belt (+0.141) · T-shirt (+0.086)
    graphical_appearance_name        Chambray (+0.062) · Embroidery (+0.047) · Other structure (+0.037)
    colour_group_name                White (+0.060) · Light Blue (+0.045) · Other Orange (+0.042)
    perceived_colour_master_name     Black (+0.053) · Unknown (+0.017) · Grey (-0.007)
    index_group_name                 Divided (+0.028) · Ladieswear (+0.026) · Sport (-0.030)
    garment_group_name               Jersey Basic (+0.094) · Shorts (+0.034) · Under-, Nightwear (+0.034)
    section_name                     Womens Everyday Collection (+0.184) · Ladies H&M Sport (+0.167) · Divided Complements Other (+0.132)
    department_name                  Price Items (+0.172) · EQ Divided Basics (+0.152) · Swimwear (+0.141)

user prototype P2:
    product_group_name               Unknown (+0.046) · Bags (+0.026) · Items (+0.022)
    product_type_name                Hair ties (+0.219) · Belt (+0.154) · T-shirt (+0.123)
    graphical_appearance_name        Chambray (+0.085) · Embroidery (+0.070) · Solid (+0.048)
    colour_group_name                White (+0.069) · Light Beige (+0.055) · Yellow (+0.052)
    perceived_colour_master_name     Black (+0.051) · Unknown (+0.010) · Grey (-0.010)
    index_group_name                 Divided (+0.032) · Ladieswear (+0.029) · Sport (-0.023)
    garment_group_name               Jersey Basic (+0.116) · Shorts (+0.033) · Under-, Nightwear (+0.032)
    section_name                     Womens Everyday Collection (+0.203) · Ladies H&M Sport (+0.181) · Divided Complements Other (+0.132)
    department_name                  Price Items (+0.184) · Swimwear (+0.174) · EQ Divided Basics (+0.158)

user prototype P3:
    product_group_name               Items (+0.026) · Unknown (+0.026) · Bags (+0.023)
    product_type_name                Hair ties (+0.203) · Belt (+0.151) · T-shirt (+0.106)
    graphical_appearance_name        Chambray (+0.077) · Embroidery (+0.043) · Solid (+0.031)
    colour_group_name                White (+0.062) · Light Beige (+0.045) · Light Blue (+0.045)
    perceived_colour_master_name     Black (+0.058) · Unknown (+0.017) · Grey (-0.006)
    index_group_name                 Ladieswear (+0.025) · Divided (+0.022) · Sport (-0.025)
    garment_group_name               Jersey Basic (+0.087) · Under-, Nightwear (+0.040) · Shorts (+0.021)
    section_name                     Womens Everyday Collection (+0.164) · Ladies H&M Sport (+0.156) · Divided Complements Other (+0.131)
    department_name                  Price Items (+0.170) · EQ Divided Basics (+0.154) · Swimwear (+0.147)
```

## 4. Worked example — exact score decomposition

```

=== EXACT decomposition: user 42 × item 0 (logit -0.0128) ===
u*ᵀt̂ half (user-prototype × item's learned affinity):
    UP2: u*=+1.697 × t̂=+0.032 = +0.0547
    UP0: u*=+1.106 × t̂=-0.049 = -0.0539
    UP3: u*=+1.046 × t̂=-0.025 = -0.0258
ûᵀt* half — ITEM-DISCRIMINATING û_k·(t*_k−1) (§3.4 amendment):
    P1: û=-0.003 × (t*−1)=+0.157 = -0.0005
        graphical_appearance_name=Solid: +0.1118
        index_group_name=Divided: +0.0164
        product_group_name=Garment Upper body: +0.0096
    P0: û=-0.001 × (t*−1)=+0.338 = -0.0005
        graphical_appearance_name=Solid: +0.1592
        product_group_name=Garment Upper body: +0.0778
        garment_group_name=Jersey Fancy: +0.0319
    P3: û=-0.003 × (t*−1)=+0.088 = -0.0002
        graphical_appearance_name=Solid: +0.0701
        product_group_name=Garment Upper body: +0.0064
        perceived_colour_master_name=Pink: +0.0034
user-constant baseline (disclosed once, ranking-irrelevant): Σû = -0.0057
SUM CHECK: Σ addends = -0.0128 vs logit -0.0128 → |err| = 9.31e-10 ✔ exact
```
