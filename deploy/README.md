# deploy/: running the dashboard as an always-on web service

## What is this folder?

This folder holds the files that turned the dashboard into a **web service** on
the original Linux server: it starts when the server starts, restarts itself
if it crashes, and can be opened by other people over an encrypted connection
after they type a password.

A few words used on this page:

- **Service:** a program that Linux keeps running in the background, with no
  terminal open.
- **systemd:** the part of Linux that starts and watches services. A
  `.service` file (a "unit file") tells it what to run, as which user and from
  which folder.
- **nginx:** a web server. Here it stands in front of the dashboard. It handles
  HTTPS (the encryption), asks for a user name and password, and then passes
  each request on to the dashboard. This arrangement is called a **reverse
  proxy**.
- **Port:** a numbered door on a computer that a program listens at. The
  dashboard uses 8501, the data server uses 8001, and HTTPS uses 443.
- **127.0.0.1 (localhost):** "this computer". A program that listens on
  127.0.0.1 cannot be reached from any other machine. Only nginx can reach it.

## Do I need it?

- **You want to look at the dashboard yourself, on your own computer:** no.
  Start it with one command instead; see [apps/README.md](../apps/README.md).
  Step 1 below shows the same command.
- **You want other people to open the dashboard from their own computers:**
  yes. You need a Linux server that uses systemd, and administrator rights on
  it (the `sudo` command).

**Who may log in.** The dashboard shows participant-level results, including
ADNI participant identifiers. ADNI data are governed by a Data Use Agreement:
you obtain them from adni.loni.usc.edu under your own agreement, and they must
never be committed to this repository. Only give a login to people covered by
such an agreement, and never make the dashboard public.

## How the pieces fit together

```text
someone's browser
     |
     |  HTTPS, port 443 (the only port open to the outside)
     v
   nginx  -- asks for user name and password --
     |
     |-- /        --> 127.0.0.1:8501  Streamlit dashboard     (connectome-dashboard.service)
     |-- /next/   --> 127.0.0.1:8001  React preview           (connectome-api.service)
     `-- /api/    --> 127.0.0.1:8001  data server (FastAPI)   (connectome-api.service)

Both services read the analysis outputs ($SC_ANALYSIS_ROOT) and the
connectome matrices ($SC_CONNECTOMES_DIR).
```

The Streamlit dashboard is the complete one. The React preview is a newer,
faster version that is still being checked; it is optional. You can deploy the
Streamlit dashboard alone.

## What's inside

| File | What it is |
|---|---|
| `README.md` | This page: how to deploy the dashboard on your own server. |
| `connectome-dashboard.service` | The systemd unit file for the Streamlit dashboard (port 8501). |
| `connectome-api.service` | The systemd unit file for the data server, which also serves the React preview (port 8001). Optional. |
| `connectome-dashboard.nginx.conf` | nginx settings for the Streamlit dashboard alone: HTTPS on 443, password, forward everything to 8501. |
| `connectome-dashboard-shadow.nginx.conf` | nginx settings for both: the Streamlit dashboard at `/`, and the React preview and data server at `/next/` and `/api/`. Use this one **or** the one above, never both. |
| `README_connectome_dashboard.md` | A short record of how the original server was set up. |
| `FASTAPI_REACT_RUNBOOK.md` | The checklist used on the original server to add the React preview next to the Streamlit dashboard, and to undo it. |

Every file here was written for the original server. The user name, the
repository folder and the Python installation inside them are that server's.
**Do not copy them into place unchanged.** Step 3 makes corrected copies for
your server.

## Before you start

1. **A Linux server with systemd,** where you can use `sudo`. The commands
   below were checked on Amazon Linux 2023. Where Debian or Ubuntu need
   something different, the page says so.
2. **The project installed,** as the top-level [README.md](../README.md)
   describes (`pip install -r requirements.txt`). That creates a **venv** (a
   private Python installation for this project) in `.venv` inside the
   repository. It also installs `fastapi` and `uvicorn`, the two packages the
   data server needs: they are listed in `requirements/dev.txt`, which
   `requirements.txt` includes. Open a terminal in the repository root (the
   folder that holds `sc_config.py`), switch the venv on, and add the
   dashboard packages (`streamlit`, `plotly`, `altair`, `scikit-image`):

   ```bash
   source .venv/bin/activate
   pip install -r requirements/dashboard.txt
   ```

   Every command on this page is run from the repository root with the venv
   on, unless it says otherwise. Check that the packages are there:

   ```bash
   python -c "import streamlit, fastapi, uvicorn; print('ok')"
   ```

   What you should see: `ok`. If it says
   `ModuleNotFoundError: No module named 'fastapi'` (or `'uvicorn'`), run
   `pip install -r requirements.txt`; if it names `'streamlit'`, run
   `pip install -r requirements/dashboard.txt`. Then check again.
3. **Your data locations set.** If you keep your `SC_*` settings in `env.sh`
   (see the top-level README), run `source env.sh` now, so that the same
   values end up in the service files. To see where every folder is:

   ```bash
   python -c "import sc_config; print(sc_config.describe())"
   ```

4. **The analysis outputs built.** The dashboard only displays results; it
   does not compute them. Build them once, as
   [connectome_analysis/README.md](../connectome_analysis/README.md) describes:

   ```bash
   python -m connectome_analysis.run_analysis --all --resume
   ```

5. **nginx, the password tool and openssl installed.** They are ordinary
   system packages:

   ```bash
   sudo dnf install -y nginx httpd-tools openssl        # Amazon Linux, Fedora, RHEL
   sudo apt install -y nginx apache2-utils openssl      # Debian, Ubuntu
   ```

6. **Node.js 18 or newer,** only if you want the React preview. Check with
   `node --version`.

## How to run it

### Step 1: start the dashboard by hand first

Before making it a service, check that it works:

```bash
python -m streamlit run apps/connectome_dashboard/connectome_app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

What you should see:

```text
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

YYYY-MM-DD hh:mm:ss.mmm Uvicorn server started on 127.0.0.1:8501

  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:8501
```

The first line is Streamlit's own notice about sending anonymous usage
statistics; the service files turn this off. The second line has today's date
and time. In a second terminal on the same machine:

```bash
curl http://127.0.0.1:8501/_stcore/health
```

What you should see: `ok`. If you have a browser on this machine, open
http://127.0.0.1:8501. The browser tab says "Connectome C Dashboard"
("Connectome C" is only the dashboard's name). The heading on the page says
"Analysis of Structural connectomes", and the first panel is "Cohort Counts"
(on the original data, "Cohort Counts — Dense cohort (n=530)"), with the groups
CN, MCI and AD. (By the project's grouping rule, EMCI, LMCI and SMC are counted
as MCI.)

Press `Ctrl+C` in the first terminal to stop it.

The Streamlit dashboard does not read `SC_ANALYSIS_ROOT` itself. It reads
`CONNECTOME_ANALYSIS_ROOT` and `CONNECTOME_MATRICES_DIR`, and falls back to
`<repo>/data/derivatives/...` when they are not set. If your data are
somewhere else, start it like this instead:

```bash
CONNECTOME_ANALYSIS_ROOT="$(python -c 'import sc_config; print(sc_config.paths().analysis_root)')" \
CONNECTOME_MATRICES_DIR="$(python -c 'import sc_config; print(sc_config.paths().connectomes_dir)')" \
python -m streamlit run apps/connectome_dashboard/connectome_app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

The service files made in Step 3 set both variables for you.

### Step 2 (optional): build the React preview

Skip this if you only want the Streamlit dashboard. The React preview is a web
page that has to be built once with Node.js. The build goes into
`apps/connectome_web/dist/`, which the data server then serves at `/next/`.

```bash
cd apps/connectome_web
npm ci
npm run build
cd ../..
```

What you should see at the end (the file names and sizes may differ a little):

```text
dist/index.html                     0.53 kB │ gzip:   0.32 kB
dist/assets/index-rcMX2TaY.css     24.87 kB │ gzip:   6.33 kB
dist/assets/index-Dcpsu-Od.js   1,735.54 kB │ gzip: 566.34 kB

(!) Some chunks are larger than 500 kB after minification. Consider:
...
✓ built in 10.13s
```

The warning about chunk size is harmless. More about this app is in
[apps/README.md](../apps/README.md).

`npm ci` may also print a `npm warn deprecated ...` line and a count of
vulnerabilities (for example `3 vulnerabilities (1 high, 2 critical)`) with the
suggestion `To address all issues, run: npm audit fix`. **Do not run
`npm audit fix`.** It rewrites `package-lock.json`, the file that pins the
exact package versions, and can break the build. The build still works
without it.

### Step 3: make your own copies of the service files

Each service file has lines that belong to the original server: the user
(`User=`, `Group=`), the repository folder (`WorkingDirectory=` and the paths
in `ExecStart=`), the data folders (`Environment=...`) and the Python
installation (the original server had one venv per app, called
`.venv_connectome_app` and `.venv_connectome_api`; you have one, `.venv`).

These commands fill in your values and write the corrected copies to a folder
called `connectome-deploy` in your home folder (`~`). They change nothing in
the repository.

```bash
REPO="$(pwd)"
ANALYSIS="$(python -c 'import sc_config; print(sc_config.paths().analysis_root)')"
MATRICES="$(python -c 'import sc_config; print(sc_config.paths().connectomes_dir)')"
OLD="$(sed -n 's/^WorkingDirectory=//p' deploy/connectome-dashboard.service)"
mkdir -p ~/connectome-deploy

sed -e "s#$OLD#$REPO#g" \
    -e "s#^User=.*#User=$(id -un)#" \
    -e "s#^Group=.*#Group=$(id -gn)#" \
    -e "s#^Environment=CONNECTOME_ANALYSIS_ROOT=.*#Environment=CONNECTOME_ANALYSIS_ROOT=$ANALYSIS\nEnvironment=CONNECTOME_MATRICES_DIR=$MATRICES#" \
    -e "s#/\.venv_connectome_app/#/.venv/#" \
    deploy/connectome-dashboard.service > ~/connectome-deploy/connectome-dashboard.service

sed -e "s#$OLD#$REPO#g" \
    -e "s#^User=.*#User=$(id -un)#" \
    -e "s#^Group=.*#Group=$(id -gn)#" \
    -e "s#^Environment=CONNECTOME_ANALYSIS_ROOT=.*#Environment=CONNECTOME_ANALYSIS_ROOT=$ANALYSIS#" \
    -e "s#^Environment=CONNECTOME_MATRICES_DIR=.*#Environment=CONNECTOME_MATRICES_DIR=$MATRICES#" \
    -e "s#/\.venv_connectome_api/#/.venv/#" \
    deploy/connectome-api.service > ~/connectome-deploy/connectome-api.service
```

What each part does:

- `REPO`, `ANALYSIS`, `MATRICES`: your repository folder and your data
  folders, as `sc_config.py` resolves them.
- `OLD`: the original server's repository folder, read from the file itself.
- Each `sed` command copies one service file and replaces the old values with
  yours. The service runs as the user you are logged in as (`id -un`).

Look at the result:

```bash
grep -E '^(User|Group|WorkingDirectory|Environment|ExecStart)=' ~/connectome-deploy/*.service
```

What you should see (with your own user name and folders in place of the
`<...>` parts):

```text
.../connectome-api.service:User=<you>
.../connectome-api.service:Group=<your group>
.../connectome-api.service:WorkingDirectory=<repo>
.../connectome-api.service:Environment=PYTHONUNBUFFERED=1
.../connectome-api.service:Environment=CONNECTOME_PROJECT_ROOT=<repo>
.../connectome-api.service:Environment=CONNECTOME_ANALYSIS_ROOT=<your SC_ANALYSIS_ROOT>
.../connectome-api.service:Environment=CONNECTOME_MATRICES_DIR=<your SC_CONNECTOMES_DIR>
.../connectome-api.service:ExecStart=<repo>/.venv/bin/uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
.../connectome-dashboard.service:User=<you>
.../connectome-dashboard.service:Group=<your group>
.../connectome-dashboard.service:WorkingDirectory=<repo>
.../connectome-dashboard.service:Environment=CONNECTOME_ANALYSIS_ROOT=<your SC_ANALYSIS_ROOT>
.../connectome-dashboard.service:Environment=CONNECTOME_MATRICES_DIR=<your SC_CONNECTOMES_DIR>
.../connectome-dashboard.service:Environment=STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
.../connectome-dashboard.service:ExecStart=<repo>/.venv/bin/streamlit run <repo>/apps/connectome_dashboard/connectome_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.enableXsrfProtection true
```

Then let systemd check the files:

```bash
systemd-analyze verify ~/connectome-deploy/*.service
```

What you should see: nothing about `connectome-dashboard.service` or
`connectome-api.service`. Lines about other, unrelated units (for example
`acpid.socket: ... legacy directory /var/run/`) can be ignored. If it says
`Command ... is not executable`, see
[If something goes wrong](#if-something-goes-wrong).

You can also open the files in a text editor (for example `nano
~/connectome-deploy/connectome-dashboard.service`) and change any line by
hand. [Settings you can change](#settings-you-can-change) explains each one.

### Step 4: install and start the services

```bash
sudo cp ~/connectome-deploy/connectome-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now connectome-dashboard.service
systemctl status connectome-dashboard --no-pager
```

`daemon-reload` makes systemd read the new file. `enable --now` starts the
service now and at every boot. What `status` should show:

```text
● connectome-dashboard.service - Connectome C Streamlit dashboard
     Loaded: loaded (/etc/systemd/system/connectome-dashboard.service; enabled; preset: disabled)
     Active: active (running) since ...
   ...
     streamlit[...]:   You can now view your Streamlit app in your browser.
     streamlit[...]:   URL: http://127.0.0.1:8501
```

Only if you built the React preview in Step 2:

```bash
sudo cp ~/connectome-deploy/connectome-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now connectome-api.service
curl http://127.0.0.1:8001/api/v1/health/live
```

What you should see:
`{"status":"ok","service":"connectome-api","schema_version":"1.0"}`

At this point both apps run, but only on 127.0.0.1. Nobody else can reach them
yet.

### Step 5: make the certificate and the password file

HTTPS needs a **certificate**: a small file that lets browsers encrypt the
connection. The command below makes a **self-signed** one, which encrypts the
connection but is not vouched for by anyone, so browsers show a warning the
first time. If your server has a domain name, you can get a certificate that
browsers trust (for example from Let's Encrypt) and put it at the same two
paths instead.

Both files go in `/etc/connectome-dashboard/`, outside the repository. Replace
`<your-server-name>` with the name or address people will type.

```bash
sudo mkdir -p /etc/connectome-dashboard/tls
sudo openssl req -x509 -newkey rsa:4096 -sha256 -nodes -days 365 \
  -keyout /etc/connectome-dashboard/tls/connectome-dashboard.key \
  -out /etc/connectome-dashboard/tls/connectome-dashboard.crt \
  -subj "/CN=<your-server-name>"
sudo chmod 600 /etc/connectome-dashboard/tls/connectome-dashboard.key
```

While it works, `openssl` prints long lines of dots, plus signs and stars
(`.+..+...+++*`).
That is normal. When it finishes, the folder holds two files:

```bash
sudo ls /etc/connectome-dashboard/tls
```

What you should see: `connectome-dashboard.crt  connectome-dashboard.key`.

The `.key` file is the secret half of the certificate. `chmod 600` makes it
readable by the administrator (root) only. The certificate lasts 365 days;
run the command again before then.

Now the password file. Replace `<username>` with the name the person will
type. `-c` creates the file, `-B` stores the password in a strong scrambled
form (bcrypt), not as plain text.

```bash
sudo htpasswd -c -B /etc/connectome-dashboard/htpasswd <username>
```

What you should see (type the password twice; nothing appears as you type):

```text
New password:
Re-type new password:
Adding password for user <username>
```

To add more people later, run the same command **without** `-c`. With `-c` the
file is replaced and everyone else's login is lost.

nginx must be able to read the password file. Find the user nginx runs as (it
is the one that is not `root`):

```bash
ps -o user= -C nginx | sort -u
```

It is usually `nginx` (Amazon Linux, Fedora, RHEL) or `www-data` (Debian,
Ubuntu). If nginx is not running yet this prints nothing; use the name for
your system. Then:

```bash
sudo chown root:nginx /etc/connectome-dashboard/htpasswd     # or root:www-data
sudo chmod 640 /etc/connectome-dashboard/htpasswd
```

### Step 6: install the nginx settings

Pick **one** of the two nginx files:

- Streamlit dashboard only:

  ```bash
  sudo cp deploy/connectome-dashboard.nginx.conf /etc/nginx/conf.d/connectome-dashboard.conf
  ```

- Streamlit dashboard **and** the React preview. This file also contains two
  `location /pe-quant` blocks that belong to a different application that
  shared the original server. The `sed` command removes them:

  ```bash
  sed -e '/location = \/pe-quant {/,/}/d' -e '/location \/pe-quant\/ {/,/}/d' \
    deploy/connectome-dashboard-shadow.nginx.conf \
    | sudo tee /etc/nginx/conf.d/connectome-dashboard.conf > /dev/null
  ```

  `sudo tee <file>` writes the file as administrator. (A plain `> <file>`
  would run as you, and you may not write into `/etc/nginx/`.) `tee` also
  prints everything it writes; `> /dev/null` throws that copy away so the
  screen stays quiet.

Both files use the certificate and password paths from Step 5, and the ports
8501 and 8001 from the service files, so nothing else has to change. The line
`server_name _;` means "answer whatever name the browser used". You may put
your server's name there instead.

Check the settings, then make nginx use them:

```bash
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

What `nginx -t` should show:

```text
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

**Fedora and RHEL only: SELinux.** SELinux is a security layer that is
switched on ("Enforcing") by default on those systems. It stops nginx from
connecting to the apps on ports 8501 and 8001, and every page then fails with
`502 Bad Gateway`. Check it with `getenforce`. If it prints `Enforcing`, allow
nginx to connect:

```bash
sudo setsebool -P httpd_can_network_connect 1
sudo systemctl reload nginx
```

`-P` keeps the setting after a restart. Amazon Linux 2023 and Debian/Ubuntu do
not enforce SELinux by default (`getenforce` prints `Permissive`, or the
command does not exist), so they do not need this. The original server did
not need it, so this step was not run there.

### Step 7: open port 443, and only port 443

The last lock is the **firewall**: the rule list that decides which computers
may connect to which ports. On the original server (an Amazon EC2 machine)
this was the "security group", and it allowed port 443 only from a short list
of single addresses (written like `203.0.113.7/32`, where `/32` means "exactly
this one address").

Do the same on your server:

- Allow port **443** only from the addresses of the people who need it.
- Never open **8501** or **8001**. The apps listen on 127.0.0.1 anyway, so an
  open rule would not reach them, but keep them closed so a later mistake in a
  service file cannot expose them.

A server can have two firewalls: one run by the cloud provider, outside the
machine, and one on the machine itself (a "host firewall"). Traffic must get
through both. Replace `203.0.113.7` below with the address of a person who
needs access, and repeat the rule for each address. (A person can find their
address by searching the web for "what is my IP address" on their own
computer.)

- **Cloud firewall** (Amazon EC2 "security group", or the same thing under
  another name at other providers): in the provider's web console, add an
  inbound rule: type HTTPS, port 443, source `203.0.113.7/32`. This is what the
  original server used.
- **Host firewall on Fedora or RHEL (`firewalld`, usually on).** Until you
  add a rule it blocks port 443, and browsers simply time out:

  ```bash
  sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="203.0.113.7/32" service name="https" accept'
  sudo firewall-cmd --reload
  ```

- **Host firewall on Ubuntu (`ufw`, off by default).** If
  `sudo ufw status` prints `Status: active`, add:

  ```bash
  sudo ufw allow from 203.0.113.7 to any port 443 proto tcp
  ```

  If it prints `Status: inactive`, there is nothing to add. Do not switch ufw
  on just for this: switching it on without also allowing SSH locks you out of
  the server.
- **Amazon Linux 2023** has no host firewall running by default, so only the
  cloud firewall applies.

These host-firewall commands are standard examples for those systems. They
were not run on the original server, which relied on the cloud firewall
alone.

### Step 8: test it

From the server itself (`-k` tells curl to accept the self-signed
certificate):

```bash
curl -k -s -o /dev/null -w "%{http_code}\n" https://localhost/
```

What you should see: `401`, which means "password required". Now with a login
(curl asks for the password):

```bash
curl -k -u <username> https://localhost/_stcore/health
```

What you should see: `ok`. If you installed the React preview, check that
the data server answers through nginx:

```bash
curl -k -u <username> https://localhost/api/v1/health/live
```

What you should see:
`{"status":"ok","service":"connectome-api","schema_version":"1.0"}`.

There is a second, stricter check, `/api/v1/health/ready`. It also looks for a
reference file that existed only on the original server (see
[If something goes wrong](#if-something-goes-wrong)), so **on your server it
always fails**:

```bash
curl -k -u <username> https://localhost/api/v1/health/ready
```

What you should see on your server (this is a `503` answer):

```text
{"detail":"required sources unavailable: ['<original repository folder>/research_audit/outputs/connectome_dashboard_reference_v1/reference.json']"}
```

This is harmless. Every other `/api/` page works without that file. The only
visible effect is that the top of the React preview keeps showing
"API checking" instead of "API ready". On the original server, where the file
existed, the answer was
`{"status":"ready","service":"connectome-api","schema_version":"1.0"}`. If the
list names any other file or folder, something is really missing; see
[If something goes wrong](#if-something-goes-wrong).

Finally, from an allowed computer, open `https://<your-server-name>/` in a
browser. Accept the certificate warning if you used a self-signed certificate,
log in, and you should see the same dashboard as in Step 1. The React preview
is at `https://<your-server-name>/next/`.

## Keeping the numbers up to date

The dashboard shows whatever is in the analysis outputs. When the matrices or
cohort tables change, rebuild the outputs and restart the services:

```bash
python -m connectome_analysis.run_analysis --all --resume
sudo systemctl restart connectome-dashboard.service
sudo systemctl restart connectome-api.service        # only if you installed it
```

The restart matters because the Streamlit dashboard keeps some results in
memory for up to an hour.

## Stopping or removing it

```bash
sudo systemctl disable --now connectome-api.service        # only if you installed it
sudo systemctl disable --now connectome-dashboard.service
sudo rm /etc/nginx/conf.d/connectome-dashboard.conf
sudo nginx -t && sudo systemctl reload nginx
```

`disable --now` stops the service and stops it starting at boot. The unit
files stay in `/etc/systemd/system/` until you delete them.

## Inputs and outputs

**Inputs** (read only, never changed):

| What | Where |
|---|---|
| Analysis outputs | `$SC_ANALYSIS_ROOT`, passed to the services as `CONNECTOME_ANALYSIS_ROOT` |
| Connectome matrices, `SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv` | `$SC_CONNECTOMES_DIR`, passed as `CONNECTOME_MATRICES_DIR` |
| Region names for the 166 matrix rows | `atlas/AAL/` in the repository |
| The built React preview | `apps/connectome_web/dist/` (Step 2) |
| Certificate and key | `/etc/connectome-dashboard/tls/connectome-dashboard.crt` and `.key` |
| Password file | `/etc/connectome-dashboard/htpasswd` |

**Outputs.** The services write no files into the repository or the data
folders. Their messages go to the system log. To read the last 50 lines:

```bash
sudo journalctl -u connectome-dashboard.service -n 50 --no-pager
sudo journalctl -u connectome-api.service -n 50 --no-pager
```

nginx keeps its own logs in `/var/log/nginx/` (`access.log` and `error.log`).

## Where passwords and keys go

**Outside the repository, always.** The password file and the certificate key
live in `/etc/connectome-dashboard/`, readable only by root (and, for the
password file, by nginx). Never copy them into the repository folder:
`.gitignore` blocks `*.key` and `*.pem` files, but **not** a file called
`htpasswd` and not the certificate, so git would happily pick them up.

If you need to write the passwords down, keep them in a password manager. The
original server kept a note of the login in a file outside the repository and
recorded its location in a variable called `CONNECTOME_CREDENTIALS_FILE`. No
program reads that variable; it was only a reminder. If you reuse any login
from the original server, change its password first.

## Settings you can change

**In the service files** (your copies in `~/connectome-deploy/`, or the
installed ones in `/etc/systemd/system/`; run `sudo systemctl daemon-reload`
and restart the service after any change):

| Line | What it means | Original value | What to put |
|---|---|---|---|
| `User=`, `Group=` | The Linux account the app runs as. | the original server's login | Your own account. Never `root`. |
| `WorkingDirectory=` | The folder the app starts in. | the original repository folder | Your repository root. |
| `Environment=CONNECTOME_PROJECT_ROOT=` (API only) | The repository root. | the original repository folder | Your repository root. |
| `Environment=CONNECTOME_ANALYSIS_ROOT=` | Where the analysis outputs are. | `<repo>/data/derivatives/qc/analysis_cohort` | Your `$SC_ANALYSIS_ROOT`. |
| `Environment=CONNECTOME_MATRICES_DIR=` | Where the connectome matrices are. The original dashboard file does not set it, so it used the default `<repo>/data/derivatives/connectomes`; Step 3 adds it. | `<repo>/data/derivatives/connectomes` | Your `$SC_CONNECTOMES_DIR`. |
| `ExecStart=` first word | The program to run, from the venv. | `.venv_connectome_app/bin/streamlit`, `.venv_connectome_api/bin/uvicorn` | `<repo>/.venv/bin/streamlit`, `<repo>/.venv/bin/uvicorn` |
| `--server.port 8501`, `--port 8001` | The port each app listens on. | 8501, 8001 | Change only together with the matching `proxy_pass` line in nginx. |
| `--server.address 127.0.0.1`, `--host 127.0.0.1` | Listen on this machine only. | 127.0.0.1 | Keep. |
| `Restart=always`, `RestartSec=5` | Restart 5 seconds after a crash. | as shown | Keep. |
| `NoNewPrivileges=true` (both files) | The app may never gain more rights than its user has. | as shown | Keep. |
| `PrivateTmp=true`, `UMask=0027` (API file only) | The data server gets its own private temporary folder, and any file it creates is not readable by other users. | as shown | Keep. |

**In the nginx file** (`/etc/nginx/conf.d/connectome-dashboard.conf`; run
`sudo nginx -t` and `sudo systemctl reload nginx` after any change):

| Line | What it means | Value in the file |
|---|---|---|
| `listen 443 ssl;` | Accept HTTPS on port 443. | 443 |
| `server_name _;` | Which host name to answer to; `_` means any. | `_` |
| `ssl_certificate`, `ssl_certificate_key` | The certificate and its key (Step 5). | `/etc/connectome-dashboard/tls/...` |
| `auth_basic "Connectome C Dashboard";` | The text shown in the login box. | as shown |
| `auth_basic_user_file` | The password file (Step 5). | `/etc/connectome-dashboard/htpasswd` |
| `proxy_pass http://127.0.0.1:8501;` | Where requests for `/` go. Must match the Streamlit port. | 8501 |
| `proxy_pass http://127.0.0.1:8001;` | Where `/next/` and `/api/` go (preview file only). Must match the API port. | 8001 |
| `proxy_read_timeout` | How long nginx waits for an answer from the app, in seconds. Streamlit keeps one connection open per open page, so `/` gets a whole day. | `86400` (one day) for `/` (Streamlit); `300` (five minutes) for `/next/` and `/api/` (preview file only) |
| `client_max_body_size 1m;` | The largest request a browser may send. | 1 MB |

## If something goes wrong

**`systemd-analyze verify` says
`connectome-dashboard.service: Command <repo>/.venv/bin/streamlit is not executable: No such file or directory`**
(or the same for `uvicorn`). The venv in that path does not exist, or the
dashboard packages are not in it. Do item 2 of
[Before you start](#before-you-start) again, then check that the file
exists with `ls <repo>/.venv/bin/streamlit`. If your venv is somewhere else,
edit the `ExecStart=` line. A service started with this mistake shows
`status=203/EXEC` in `systemctl status`.

**`systemctl status` shows `status=217/USER`.** The `User=` or `Group=` line
names an account that does not exist on this server. Put your own account
there (Step 3 does this for you), then `sudo systemctl daemon-reload` and
restart.

**The browser shows `Analysis root not found: <path>`.** The Streamlit
dashboard started, but `CONNECTOME_ANALYSIS_ROOT` points to a folder that does
not exist. Either the analysis has not been run yet (item 4 of
[Before you start](#before-you-start)), or the path in the service file is
wrong. Fix it, run `sudo systemctl daemon-reload`, and restart the service.

**`ModuleNotFoundError: No module named 'fastapi'`** (or `'uvicorn'`), in the
package check of [Before you start](#before-you-start) or in
`journalctl -u connectome-api.service`. The data server's packages are not in
the venv. They come from the top-level install: run
`pip install -r requirements.txt` with the venv on. For
`No module named 'streamlit'`, run `pip install -r requirements/dashboard.txt`.
Then restart the service.

**`Port 8501 is not available`** when starting the dashboard by hand. Something
already listens on that port, often the service you installed. See what with
`ss -ltn | grep ':8501 '`. Stop the service
(`sudo systemctl stop connectome-dashboard.service`) or use another port for
the test, for example `--server.port 8502`.

**`sudo nginx -t` says
`[emerg] cannot load certificate "/etc/connectome-dashboard/tls/connectome-dashboard.crt": BIO_new_file() failed (... No such file or directory ...)`.**
The certificate is missing or somewhere else. Do Step 5, or correct the
`ssl_certificate` lines.

**`sudo nginx -t` says `[warn] conflicting server name "_" on 0.0.0.0:443, ignored`.**
Both nginx files are installed, so two sets of settings claim port 443. Keep
only one file in `/etc/nginx/conf.d/`.

**The browser or curl says `401 Authorization Required` even with the right
password.** The user name or password does not match the password file. The
nginx error log says which: `sudo tail /var/log/nginx/error.log` shows
`user "<username>": password mismatch` or
`user "<username>" was not found in "/etc/connectome-dashboard/htpasswd"`.
Set the password again with
`sudo htpasswd -B /etc/connectome-dashboard/htpasswd <username>` (no `-c`).

**`403 Forbidden` for everyone.** The password file does not exist. The error
log shows
`open() "/etc/connectome-dashboard/htpasswd" failed (2: No such file or directory)`.
Create it as in Step 5.

**`500 Internal Server Error` for everyone.** The password file exists, but
nginx may not read it. The error log shows
`open() "/etc/connectome-dashboard/htpasswd" failed (13: Permission denied)`.
Fix its owner and permissions with the `chown` and `chmod` commands in Step 5.

**`502 Bad Gateway`.** nginx works, but it cannot reach the app behind it.
Look at the end of the error log (`sudo tail /var/log/nginx/error.log`):

- `connect() failed (111: Connection refused) while connecting to upstream`:
  the app is not running. Check the service with
  `systemctl status connectome-dashboard --no-pager` (or `connectome-api`) and
  read why it stopped with
  `sudo journalctl -u connectome-dashboard.service -n 50 --no-pager`.
- `(13: Permission denied) while connecting to upstream`: the app is running,
  but SELinux (Fedora, RHEL) does not let nginx connect to it. Run
  `sudo setsebool -P httpd_can_network_connect 1`, then
  `sudo systemctl reload nginx`. See the end of Step 6.

**The page never loads and the browser finally says the connection timed
out.** A firewall blocks port 443 before the request reaches nginx, so nothing
appears in the nginx logs. Check both firewalls as in Step 7, and that the
address you are connecting from is on the list.

**`curl: (60) SSL certificate problem: self-signed certificate`.** Expected
with a self-signed certificate. Add `-k` to curl. Browsers show a warning page
instead; accept it, or use a trusted certificate (Step 5).

**`/api/v1/health/ready` answers `503` with
`{"detail":"required sources unavailable: [...]"}`.** The list names the files
and folders the data server looked for and could not find:

- `.../research_audit/outputs/connectome_dashboard_reference_v1/reference.json`:
  expected on every server except the original one. This check uses a fixed
  path to a reference file on the original server (`REFERENCE_FILE` in
  `apps/connectome_api/main.py`), and that file is not part of the
  repository. It is harmless: the other `/api/` pages do not need it. The only
  effect is that the top of the React preview keeps showing "API checking"
  instead of "API ready".
- `.../00_master/master_cohort.csv`: the analysis has not been run, or
  `CONNECTOME_ANALYSIS_ROOT` is wrong. Fix it as for "Analysis root not found"
  above, then `sudo systemctl restart connectome-api.service`.
- your matrices folder (for example `<repo>/data/derivatives/connectomes`):
  `CONNECTOME_MATRICES_DIR` is wrong, or the matrices are not there. Fix the
  `Environment=CONNECTOME_MATRICES_DIR=` line in the installed
  `connectome-api.service`, then run `sudo systemctl daemon-reload` and
  `sudo systemctl restart connectome-api.service`.

**`https://<your-server-name>/next/` shows `{"detail":"Not Found"}`.** The
React preview was not built, so the data server has nothing to show. Do Step 2,
then `sudo systemctl restart connectome-api.service` (the server only looks
for the build when it starts).

**The "Live Pipeline Monitor" view of the Streamlit dashboard shows warnings.**
That view watched the imaging runs on the original server and looks for files
that only existed there. Switch back to the "Analysis" view, which has all the
results. The switch is in the sidebar, which starts closed: click the small
arrow at the top left of the page to open it, and under **View** choose the
first option, **Analysis**. (Both options start with a small picture before
the word.)

## More detail

- [apps/README.md](../apps/README.md): what each app is and how to run it by
  hand.
- [README_connectome_dashboard.md](README_connectome_dashboard.md): how the
  original server was set up.
- [FASTAPI_REACT_RUNBOOK.md](FASTAPI_REACT_RUNBOOK.md): how the React preview
  was added next to the Streamlit dashboard on the original server, the tests
  run before and after, and how to undo it. Its paths are the original
  server's.
