* generate_fixtures.do — Run in Stata to produce parity fixture CSVs
* Usage: cd tests/fixtures/stata && stata -b do generate_fixtures.do

clear all
set more off

local base_dir = "`c(pwd)'"

* Load data
import delimited "`base_dir'/../parity_data.csv", clear

* Helper program to export results
capture program drop export_results
program define export_results
    args filename
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    local k = colsof(b)
    local n = e(N)

    tempname fh
    file open `fh' using "`filename'", write replace
    file write `fh' "variable,coef,se,t,p" _n
    forvalues i = 1/`k' {
        local name : word `i' of `names'
        local coef = b[1, `i']
        local se = sqrt(V[`i', `i'])
        local t = `coef' / `se'
        local p = 2 * ttail(e(df_r), abs(`t'))
        file write `fh' "`name',`coef',`se',`t',`p'" _n
    }
    * Add stats row
    file write `fh' "_stat_n,`n',,,," _n
    capture local r2 = e(r2)
    if _rc == 0 {
        file write `fh' "_stat_r2,`r2',,," _n
    }
    capture local f = e(F)
    if _rc == 0 {
        file write `fh' "_stat_F,`f',,," _n
    }
    file close `fh'
end

local outdir "`base_dir'"

* ─── OLS ───
reg y x1 x2
export_results "`outdir'/ols_iid.csv"

reg y x1 x2, vce(hc2)
export_results "`outdir'/ols_hc2.csv"

reg y x1 x2, vce(hc3)
export_results "`outdir'/ols_hc3.csv"

reg y x1 x2, vce(robust)
export_results "`outdir'/ols_hc1.csv"

reg y x1 x2, vce(cluster firm_id)
export_results "`outdir'/ols_cluster.csv"

newey y x1 x2, lag(4)
export_results "`outdir'/ols_nw.csv"

* DK requires xtset
xtset firm_id year_id
xtscc y x1 x2, lag(4)
export_results "`outdir'/ols_dk.csv"

* ─── OLS + FE ───
reghdfe y x1 x2, absorb(firm_id) vce(cluster firm_id)
export_results "`outdir'/ols_fe_cluster.csv"

reghdfe y x1 x2, absorb(firm_id) vce(robust)
export_results "`outdir'/ols_fe_hc1.csv"

reghdfe y x1 x2, absorb(firm_id year_id) vce(cluster firm_id)
export_results "`outdir'/ols_2fe_cluster.csv"

* ─── 2SLS ───
ivregress 2sls y x1 (x_endog = z1 z2)
export_results "`outdir'/iv_iid.csv"

ivregress 2sls y x1 (x_endog = z1 z2), vce(robust)
export_results "`outdir'/iv_robust.csv"

ivregress 2sls y x1 (x_endog = z1 z2), vce(cluster firm_id)
export_results "`outdir'/iv_cluster.csv"

* IV + FE
ivreghdfe y x1 (x_endog = z1 z2), absorb(firm_id) cluster(firm_id)
export_results "`outdir'/iv_fe_cluster.csv"

* ─── Panel RE ───
xtset firm_id year_id
xtreg y x1 x2, re
export_results "`outdir'/re_iid.csv"

xtreg y x1 x2, re vce(cluster firm_id)
export_results "`outdir'/re_cluster.csv"

* ─── Newey (baseline HAC check) ───
newey y x1 x2, lag(8)
export_results "`outdir'/newey_lag8.csv"

di "All fixtures generated."
