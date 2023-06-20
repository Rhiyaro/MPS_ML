
def check_for_channel(data, channel):
    if channel not in data.columns:
        raise Exception(f'Channel {channel} absent from the dataset')


def check_amount_of_data(data, min_amount):
    if data.shape[0] < min_amount:
        if data.shape[0] == 0:
            raise Exception('No data available')
        else:
            raise Exception('Too little data available')


def check_model_compatibility_classifier(turbine_reg_id: int,
                                         turbine_model_reg_ids: list[int],
                                         loader):

    turbine_model_reg_id = loader._load_turbine_model(turbine_reg_id)

    if turbine_model_reg_id in turbine_model_reg_ids:
        return True

    return False
