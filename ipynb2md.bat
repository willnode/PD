@echo off
cd src
for %%f in (*.ipynb) do jupyter nbconvert --output-dir="../docs" --to markdown "%%f"
cd DPWD
for %%f in (*.ipynb) do jupyter nbconvert --output-dir="../../docs/DPWD" --to markdown "%%f"
cd ..\..