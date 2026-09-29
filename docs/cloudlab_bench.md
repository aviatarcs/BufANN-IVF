# Isolated benchmarks on CloudLab

This is the runbook for a Claude session that manages BufANN-IVF benchmark
experiments on CloudLab. You work in the CloudLab portal
(https://www.cloudlab.us) through Chrome and on the nodes over SSH.

The goal is to run several benchmarks at once without them affecting each
other. Each run gets its own experiment, and so its own bare-metal node, all
booted from one disk image. On a shared node the runs would compete for
cores, memory bandwidth, the L3 cache, the page cache and the disks, and
core pinning does not separate the last four.

You need these three things. If any is missing, stop and ask the human:

- A CloudLab login in Chrome, as a member of project `ut-data`.
- A shell that can `ssh` to CloudLab nodes with the human's key. The portal
  shows each node's `ssh user@host` line on the experiment's "List View" tab.
- A list of the runs to do. Each run is a git ref plus a benchmark command
  (see "The runs").

## Rules

- **Only touch experiments you created**, and name every one `ivfb-<what>-<n>`.
  `evpeng-bufann` is the human's development node, where other agents also
  work. Never terminate, extend, reboot or snapshot it without the human's
  explicit go-ahead for that specific action. The project has other users,
  so leave their experiments and datasets alone.
- **Never mount the project's `fsnode` dataset** (mounted at `/local/data`
  on `evpeng-bufann`). It is another user's read-write dataset, and a
  read-write mount locks everyone else out of it.
- **Use one hardware type for everything you compare.** The default is
  `sm110p` at Wisconsin, the development node's type. If no `sm110p` nodes
  are free, do not substitute another type: tell the human when they will be
  free (see the portal's Resource Availability page) and wait.
- **Run one benchmark per node at a time.** Nothing else may run on a bench
  node while it measures, including another Claude session.
- **Copy the results off before terminating**, then terminate. Experiments
  hold scarce nodes, and their scratch disk is wiped at termination. Extend
  an experiment only if a run genuinely needs the time.
- **Keep a ledger** at `/tmpdata/ivf_bench/cloudlab/LEDGER.md` on the
  development node. Give each experiment one line: name, node, ref, command,
  status (created / preparing / running / collected / terminated) and a
  note. Update it at every state change. After losing context or restarting,
  read it first, so no experiment is left running forgotten.

## One-time setup

Check the portal first; this may already be done. You are looking for an
image named `bufann-ivf-bench`, a dataset named `bufann-ivf-data` and a
profile named `bufann-ivf-bench` in project `ut-data`. Ask the human before
each of these steps.

1. **Disk image.** On `evpeng-bufann`'s node, use "Create Disk Image" and
   name it `bufann-ivf-bench`. The image contains the root filesystem only:
   Ubuntu 24.04, the toolchain, and the hand-installed
   `/lib/x86_64-linux-gnu/libmkl_rt.so` that `dev_env.sh` links against.
   **Snapshotting takes the node offline for several minutes and kills
   everything running on it**, including tmux sessions and other Claude
   sessions. Do it only when the human says the node is free.
2. **Dataset `bufann-ivf-data`.** It holds about 15 GB, laid out as:

   ```
   SHA256SUMS                         sha256sum of every file, relative paths
   sift10m/base.9M.bin                first 9M rows of /tmpdata/sift10m/base.10M.bin
   sift10m/query.10K.bin
   sift10m/sift10m_9000000_gt10.bin
   sift100m/base.90M.bin
   sift100m/query.10K.bin
   sift100m/sift100m_90000000_gt100.bin
   ```

   The files currently live on `evpeng-bufann`:

   - `sift10m/base.9M.bin` is at `/tmpdata/ivf_bench/sift10m/base.9M.bin`.
   - `sift10m/query.10K.bin` is at `/tmpdata/sift10m/query.10K.bin`.
   - `sift10m/sift10m_9000000_gt10.bin` is at
     `/tmpdata/BufANN-CS395T/eval_tempfiles/sift10m/DiskANN/query/groundtruth/sift10m_9000000_gt10.bin`.
   - The three `sift100m/` files are in `/tmpdata/sift100m/`.

   Choose one of these two dataset kinds:

   - **Image-backed (preferred).** CloudLab copies it onto each node's local
     disk when the experiment starts, so benchmarks never touch the network,
     and any number of experiments can use it at once.
   - **Long-term remote.** A remote dataset mounted read-only can be used by
     several experiments at once. `cloudlab_node_prep.sh` copies it to local
     disk before building.

   Either way, you fill it once, in a single-node experiment `ivfb-stage`
   that has scratch space. The development node's `/tmpdata` is 97% full, so
   do not stage there. Copy the files in through your own machine with
   `scp -3 <dev>:<path> <stage>:<path>`, since the nodes hold no keys for
   each other. Then run `sha256sum` to write `SHA256SUMS`, check it against
   the same checksums computed on the development node, and create the
   dataset from that blockstore. CloudLab's "Datasets" documentation has the
   exact portal steps. Terminate `ivfb-stage` when you are done.
3. **Profile `bufann-ivf-bench`.** Create it in the portal from the contents
   of `cloudlab/profile.py`. Its parameters are:
   - `image`: the image URN.
   - `hwtype`: `sm110p`.
   - `dataset`: the dataset URN.
   - `dataset_kind`: `image` or `remote`.
   - `scratch_gb`: 400 GB, enough for a 90M index.

   The profile has not been instantiated yet. If the portal rejects it, fix
   `profile.py` in a commit.

## The runs

Before starting anything, write the batch into the ledger, one line per run:

- **Name**, for example `ivfb-nlist-32k-1`.
- **Git ref**: a branch or commit of `aviatarcs/BufANN-IVF`. The query
  bench and `nlist_sweep.sh` are on `agent/work`, not `main`.
- **Command**, run from `/tmpdata/BufANN-IVF`. Dataset files are under
  `/tmpdata/data`: the prep script copies `/dataset` there, because under
  this profile `/dataset` is always a separate volume. Results go to
  `/tmpdata/results`. For example:

  ```
  scripts/nlist_sweep.sh uint8 /tmpdata/data/sift10m/base.9M.bin /tmpdata/data/sift10m/query.10K.bin \
      /tmpdata/data/sift10m/sift10m_9000000_gt10.bin /tmpdata/results 32768
  ```

For comparisons, use one run per configuration and include the baseline
configuration as a run of its own. Nodes of the same type still differ by a
few percent. If two configurations are within that spread, run each on
every node, or run them again on swapped nodes, before calling a
difference. Report the spread across nodes, not only the means.

## Per-run workflow

1. **Start the experiment.** Go to Experiments, then Start Experiment, and
   choose profile `bufann-ivf-bench`. Set the parameters, name the
   experiment after the run, pick the Wisconsin cluster, and ask for the
   run's expected time plus about 50%. Add a ledger line with status
   `created`.
2. **Wait until the portal shows it Ready.** This usually takes 10–20
   minutes, and longer while an image-backed dataset loads. Check every few
   minutes. If it fails to start for lack of nodes, record that in the
   ledger and try again later; do not change the hardware type.
3. **Prepare the node and start the run in tmux**, so that an SSH drop does
   not kill it. Write the run's ref and command literally in place of
   `<ref>` and `<command>`:

   ```
   ssh <node> 'sudo chown $USER /tmpdata && mkdir -p /tmpdata/results &&
     curl -fsSL https://raw.githubusercontent.com/aviatarcs/BufANN-IVF/main/scripts/cloudlab_node_prep.sh -o /tmpdata/prep.sh &&
     chmod +x /tmpdata/prep.sh'
   ssh <node> 'cat > /tmpdata/run.sh' <<'EOF'
   #!/bin/bash
   { /tmpdata/prep.sh /dataset <ref> && cd /tmpdata/BufANN-IVF && <command>; } > /tmpdata/results/run.log 2>&1
   echo run_exit=$? >> /tmpdata/results/run.log
   EOF
   ssh <node> 'chmod +x /tmpdata/run.sh && tmux new -d -s bench /tmpdata/run.sh'
   ```

   The prep script is fetched from `main` and run as a copy, because it
   checks out `REF`, which may not contain it. It then does the following,
   stopping with `PREP FAILED: <reason>` at the first problem:
   - checks the node type;
   - checks out `REF`;
   - stages the data and verifies `SHA256SUMS`;
   - builds and runs `ctest`;
   - waits for the load average to drop below 1;
   - writes `/tmpdata/results/node.json`, recording the node, CPU, kernel,
     governor, commit and data checksums.

   Set the ledger status to `preparing`.
4. **Watch the run** by checking `tail -3 /tmpdata/results/run.log` every
   10–30 minutes, depending on how long the run takes. Do not poll
   continuously, and never run anything else on the node. Set the status to
   `running` once `READY` appears, and note `run_exit=` when it appears.
   If you see `PREP FAILED`, see Troubleshooting.
5. **Collect the results** with
   `scp -3 -r <node>:/tmpdata/results <dev>:/tmpdata/ivf_bench/cloudlab/<experiment>/`.
   This copies logs and JSON only; indexes are large and can be rebuilt.
   Confirm that `node.json` is there, that `run_exit=0`, and that
   `node.json`'s `nodetype` and `commit` are what the ledger says. Set the
   status to `collected`.
6. **Terminate the experiment** in the portal and set the status to
   `terminated`.
7. **Report back to the human** with one table covering the whole batch
   (run, node, commit, headline numbers, spread across nodes) and the path
   to the collected results. Recall at a fixed configuration and index is
   deterministic, so it must match across nodes. A mismatch means the runs
   did not use the same inputs. Say so rather than averaging over it.

## Troubleshooting

- **`node type is X, expected sm110p`**: the profile's `hwtype` was not
  set. Terminate the experiment and start it again.
- **`no /lib/x86_64-linux-gnu/libmkl_rt.so`**: the node did not boot the
  bench image. Check the profile's `image` parameter.
- **`checksum mismatch`**: the dataset or the copy of it is damaged. Do not
  benchmark. Rerun the prep once, and if it fails again, report it.
- **`build or ctest failed`**: `REF` is broken. The run is invalid; report
  it along with the tail of `run.log`.
- **`node not idle`**: something else is using the node. `run.log` lists the
  top processes. Find out what is running before rerunning.
- **Disk full**: raise `scratch_gb`. A 90M index takes about 20 GB, and a
  data copy from a remote dataset takes another 15 GB.
