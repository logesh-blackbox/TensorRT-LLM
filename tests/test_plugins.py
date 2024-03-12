# Test loading the TensorRT-LLM plugin library
def test_load_library():
    """Test loading the TensorRT-LLM plugin library."""
    runtime = _trt.Runtime(_trt.Logger(_trt.Logger.WARNING))
    registry = runtime.get_plugin_registry()
    handle = registry.load_library(_tlp.plugin_lib_path())
    creators = registry.plugin_creator_list
    # Check if the number of creators is greater than or equal to 10
    assert len(creators) >= 10
    # Iterate through the creators and check if the plugin namespace is correct
    for creator in creators:
        assert creator.plugin_namespace == _tlp.TRT_LLM_PLUGIN_NAMESPACE

    # Deregister the library and check if the plugin creator list is empty
    registry.deregister_library(handle)
    assert len(registry.plugin_creator_list) == 0
