import React, {useState} from "react";
import {Panel, SelectPicker} from "rsuite";


interface SelectModelProps {
    models: string[],
    selected: string,
    onChange: () => void
    setSelectedModel: (val: string) => void
}

const SelectModel = ({models, selected, onChange, setSelectedModel}: SelectModelProps) => {
    return (
        <Panel shaded header={'Выберите область анализа'}>
            <SelectPicker value={selected}
                          data={models.map(m => ({
                              label: m,
                              value: m
                          }))}
                          placeholder={'Выбрать'}
                          onChange={v => {
                              setSelectedModel(v);
                              onChange()
                          }}
            />
        </Panel>
    );
}

export default SelectModel;