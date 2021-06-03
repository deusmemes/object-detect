import React, {useState} from "react";
import styles from './StepsBlock.module.scss'
import {Steps} from "rsuite";

interface StepsBlockProps {
    step: number
}

const StepsBlock = ({step} : StepsBlockProps) => {
    return (
        <div className={styles.steps}>
            <Steps current={step} vertical>
                <Steps.Item title="Область" description="Модель"/>
                <Steps.Item title="Изображения" description="Загрузка"/>
                <Steps.Item title="Результаты" description="Динамика"/>
            </Steps>
        </div>
    )
}

export default StepsBlock;