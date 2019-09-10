cd src
for %%f in (*.ipynb) do jupyter nbconvert --output-dir="../docs" --to markdown %%f
cd ..