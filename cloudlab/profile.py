"""One bare-metal node for an isolated BufANN-IVF benchmark run.

Boots the bench disk image on a fixed hardware type, gives it local scratch
at /tmpdata, and attaches the benchmark datasets read-only at /dataset.
Instructions: docs/cloudlab_bench.md in https://github.com/aviatarcs/BufANN-IVF.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
pc.defineParameter("image", "Disk image URN (the bufann-ivf-bench snapshot)",
                   portal.ParameterType.IMAGE, "")
pc.defineParameter("hwtype", "Hardware type; every node in a comparison must match",
                   portal.ParameterType.NODETYPE, "sm110p")
pc.defineParameter("dataset", "Dataset URN (bufann-ivf-data)",
                   portal.ParameterType.STRING, "")
pc.defineParameter("dataset_kind", "Dataset kind",
                   portal.ParameterType.STRING, "image",
                   [("image", "image-backed (loaded onto local disk)"),
                    ("remote", "long-term remote (iSCSI)")])
pc.defineParameter("scratch_gb", "Local scratch at /tmpdata (GB), on the NVMe drives",
                   portal.ParameterType.INTEGER, 3000)
params = pc.bindParameters()
if not params.image:
    pc.reportError(portal.ParameterError("set the bench image URN", ["image"]))
if not params.dataset:
    pc.reportError(portal.ParameterError("set the dataset URN", ["dataset"]))
pc.verifyParameters()

request = pc.makeRequestRSpec()
node = request.RawPC("node")
node.hardware_type = params.hwtype
node.disk_image = params.image

# Off the system disk: on sm110p that is a SATA SSD, and a volume spanning
# it and the NVMe drives makes I/O timings depend on extent placement.
scratch = node.Blockstore("scratch", "/tmpdata")
scratch.size = "%dGB" % params.scratch_gb
scratch.placement = "nonsysvol"

if params.dataset_kind == "image":
    data = node.Blockstore("dataset", "/dataset")
    data.dataset = params.dataset
    data.placement = "nonsysvol"
else:
    iface = node.addInterface("if-dataset")
    remote = request.RemoteBlockstore("dsnode", "/dataset")
    remote.dataset = params.dataset
    remote.readonly = True  # read-only is what lets several experiments mount it at once
    link = request.Link("dataset-link")
    link.addInterface(iface)
    link.addInterface(remote.interface)
    link.best_effort = True
    link.vlan_tagging = True

pc.printRequestRSpec(request)
