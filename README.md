Read and plot CISMIP formatted v2 instrumented building earthquake motion records such as posted on CESMD website (Center of Strong Motion Data https://www.strongmotioncenter.org/). The code can also be used to view building instrument records on the CESMD site and those also available at the HCAI website (https://hcai.ca.gov/construction-finance/facility-detail/ - navigate to a hospital that has instrumented buildings and look under the Instrumented Buildings Tab). Python code reads a zip file containing .v2 files that contains multiple channels (Instrumented Buildings have records for individual channels zipped together)

3/17/2025 Absolute trigger for autoranging was failing for earthquake records that have very low accelerations - revised for a dyanmic trigger based on 1/10 maximum absolute value.


Try it at https://appinstrumentedbldgs-mltednwkb9re4umeqmdwkb.streamlit.app/
