@echo off
set "DATA_DIR=C:\Users\viktor.jonsson\OneDrive - CellMax Technologies AB\Documents\Repositories\svenviktorjonsson\beach-tracker\data"
cd /d "C:\Users\viktor.jonsson\OneDrive - CellMax Technologies AB\Documents\Repositories\svenviktorjonsson\beach-tracker"
"C:\Python\Python313\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8081 1>>"C:\Users\viktor.jonsson\OneDrive - CellMax Technologies AB\Documents\Repositories\svenviktorjonsson\beach-tracker\tmp-labeler.log" 2>>"C:\Users\viktor.jonsson\OneDrive - CellMax Technologies AB\Documents\Repositories\svenviktorjonsson\beach-tracker\tmp-labeler.err.log"
