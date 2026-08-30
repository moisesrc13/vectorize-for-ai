import torch._dynamo

# Must be the very first torch configuration, before any model (docling, HuggingFace,
# etc.) is imported. TorchDynamo/Inductor compilation causes a SymPy BooleanAtom
# crash with the transformer models used in this package. Eager mode is correct
# and fast enough for these small encoder / layout models.
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
