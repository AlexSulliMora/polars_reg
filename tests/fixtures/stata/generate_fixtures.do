* generate_fixtures.do — Run in Stata to produce parity fixture CSVs
* Usage: cd tests/fixtures/stata && stata -b do generate_fixtures.do
*
* Required packages (install once):
*   ssc install reghdfe
*   ssc install ftools
*   ssc install ivreghdfe
*   ssc install ivreg2
*   ssc install ranktest
*   ssc install xtscc

clear all
set more off

local base_dir = "`c(pwd)'"

* Load data (500 firms x 20 years, pre-sorted by firm then year)
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
    file write `fh' "_stat_n,`n',,," _n
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

* ─── HAC / Newey-West ───
* newey with panel data may not work in StataBE.
* Test on a single firm's time series instead.
preserve
keep if firm_id == 1
tsset year_id
capture noisily newey y x1 x2, lag(4)
if _rc == 0 {
    export_results "`outdir'/ols_nw.csv"
}
else {
    di "SKIPPED: newey lag(4) failed"
}
restore

* DK requires xtscc package + panel setup
preserve
sort firm_id year_id
by firm_id: gen t = _n
capture tsset firm_id t
capture noisily xtscc y x1 x2, lag(4)
if _rc == 0 {
    export_results "`outdir'/ols_dk.csv"
}
else {
    di "SKIPPED: xtscc not available or panel setup failed"
}
restore

* ─── OLS + FE (requires reghdfe) ───
capture noisily reghdfe y x1 x2, absorb(firm_id) vce(cluster firm_id)
if _rc == 0 {
    export_results "`outdir'/ols_fe_cluster.csv"
}

capture noisily reghdfe y x1 x2, absorb(firm_id) vce(robust)
if _rc == 0 {
    export_results "`outdir'/ols_fe_hc1.csv"
}

capture noisily reghdfe y x1 x2, absorb(firm_id year_id) vce(cluster firm_id)
if _rc == 0 {
    export_results "`outdir'/ols_2fe_cluster.csv"
}

* ─── 2SLS ───
capture noisily ivregress 2sls y x1 (x_endog = z1 z2)
if _rc == 0 {
    export_results "`outdir'/iv_iid.csv"
}

capture noisily ivregress 2sls y x1 (x_endog = z1 z2), vce(robust)
if _rc == 0 {
    export_results "`outdir'/iv_robust.csv"
}

capture noisily ivregress 2sls y x1 (x_endog = z1 z2), vce(cluster firm_id)
if _rc == 0 {
    export_results "`outdir'/iv_cluster.csv"
}

* IV + FE (requires ivreghdfe)
capture noisily ivreghdfe y x1 (x_endog = z1 z2), absorb(firm_id) cluster(firm_id)
if _rc == 0 {
    export_results "`outdir'/iv_fe_cluster.csv"
}

* ─── Panel RE ───
preserve
sort firm_id year_id
by firm_id: gen t = _n
capture tsset firm_id t
if _rc == 0 {
    capture noisily xtreg y x1 x2, re
    if _rc == 0 {
        export_results "`outdir'/re_iid.csv"
    }

    capture noisily xtreg y x1 x2, re vce(cluster firm_id)
    if _rc == 0 {
        export_results "`outdir'/re_cluster.csv"
    }
}
else {
    di "SKIPPED: panel setup failed for xtreg"
}
restore

* ─── Newey lag 8 (baseline HAC check on single firm) ───
preserve
keep if firm_id == 1
tsset year_id
capture noisily newey y x1 x2, lag(8)
if _rc == 0 {
    export_results "`outdir'/newey_lag8.csv"
}
else {
    di "SKIPPED: newey lag(8) failed"
}
restore

di "All fixtures generated."
