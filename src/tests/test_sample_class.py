from src.sample_class import SampleClass


sample = SampleClass(1)


def test_sample_class_d0_something2():
    # sample = SampleClass(1)
    assert sample.do_something2('a', 'b') == 1
    print('ok')


# pytest should be run in the console at the main project folder
# (the parent of src)
# write this in the console and run:
# pytest --cov
# Stuff can be defined outside of the scope of the functions if needed
