-- Drop every slide whose title starts with "Backup" when rendered with
--   -M hide-appendix:true
-- (used by the GitHub Pages workflow; local renders keep the backup slides).

function Pandoc(doc)
  local flag = doc.meta["hide-appendix"]
  if not (flag == true or (flag and pandoc.utils.stringify(flag) == "true")) then
    return doc
  end

  local out, skipping = {}, false
  for _, blk in ipairs(doc.blocks) do
    if blk.t == "Header" and blk.level <= 2 then
      skipping = pandoc.utils.stringify(blk.content):match("^Backup") ~= nil
    end
    if not skipping then table.insert(out, blk) end
  end
  doc.blocks = out
  return doc
end
