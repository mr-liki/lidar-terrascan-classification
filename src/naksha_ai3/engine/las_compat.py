"""LAS compatibility installed independently in the isolated AI interpreter."""


def install_opaque_extra_bytes_fix():
    # LAS data_type=0 uses options as the byte count. laspy otherwise reads
    # bits in counts such as 21 as scale/offset flags, creating invalid arrays.
    from laspy.vlrs.known import ExtraBytesStruct

    if getattr(ExtraBytesStruct, "_naksha_opaque_fixed", False):
        return

    def opaque_safe(prop):
        def read(self):
            return None if self.data_type == 0 else prop.fget(self)
        return property(read, prop.fset, prop.fdel, prop.__doc__)

    for name in ("scale", "offset", "no_data"):
        setattr(ExtraBytesStruct, name, opaque_safe(getattr(ExtraBytesStruct, name)))
    ExtraBytesStruct._naksha_opaque_fixed = True


install_opaque_extra_bytes_fix()
