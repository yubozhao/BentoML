import * as React from "react";
import { Tag, Intent } from "@blueprintjs/core";

const DeploymentStatusTag = ({state}) => {
  let statusColor;
  switch (state) {
    case "RUNNING":
    case "SUCCESSED":
      statusColor = Intent.SUCCESS;
      break;
    case "FAILED":
    case "ERROR":
    case "CRASH_LOOP_BACK_OFF":
      statusColor = Intent.DANGER;
      break;
    case "PENDING":
    default:
      statusColor = Intent.NONE;
  }

  return <Tag intent={statusColor}>{state}</Tag>
};

export default DeploymentStatusTag;