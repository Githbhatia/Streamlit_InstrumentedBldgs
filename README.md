Read and plot CESMD/CISMIP formatted v2 instrumented building earthquake motion records such as posted on CESMD website (Center of Strong Motion Data https://www.strongmotioncenter.org/). The code can also be used to view building instrument records on the CESMD site and those also available at the HCAI website (https://hcai.ca.gov/construction-finance/facility-detail/ - navigate to a hospital that has instrumented buildings and look under the Instrumented Buildings Tab). Python code reads a zip file containing .v2 files that contains multiple channels (Instrumented Buildings have records for individual channels zipped together)

3/17/2025 Absolute trigger for autoranging was failing for earthquake records that have very low accelerations - revised for a dyanmic trigger based on 1/10 maximum absolute value.

4/16/2025 CSMIP has changed the format files available, revised to read all channels from a single v2 file as in the new format files.

4/16/2025 Corrected error when only a single vertical channel is present, as a result some dead space is added to vertical plot space (yet to figure out how to remove the dead space).

4/18/2025 Added option to change damping ratio and end-period for floor spectra.  Also, added an option to rearrange the channels by dragging and dropping using streamlit_sortables (which is pretty neat).

5/3/2025 Added option to calculate drift from user selected channels (further refinement needed).  Also added option to output velocity and displacement channels.

5/11/2025 Added option to calculate resultant drift  (4 Channels over 2 storeys). Corrected issue with channels being jumbled when rearranged.

7/18/2025 Revised to read older format files, revised to adjust for unequal records (some older files have unequal records in some channels).

Try it at https://appinstrumentedbldgs-mltednwkb9re4umeqmdwkb.streamlit.app/
