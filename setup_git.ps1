# Run this from inside the vitaliy-kronos folder
# Option A: Right-click the folder -> "Open Git Bash here" -> paste commands below
# Option B: Open PowerShell in this folder and run this script

git init -b main
git config user.email "pikalo.vitaliy@gmail.com"
git config user.name "Vitaliy-Pikalo"
git add .
git commit -m "initial commit: ICT session backtester + Kronos AI filter"
git remote add origin https://github.com/Vitaliy-Pikalo/vitaliy-kronos.git
git push -u origin main
